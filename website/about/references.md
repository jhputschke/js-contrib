# Publications and citing

If you publish results made with js-contrib, cite the repository and the papers of the physics
modules your results depend on. This page lists them by stage of the simulation, says which
ones belong to which production, and gives a citation for js-contrib itself. Every arXiv
number below links to the paper's abstract page. For BibTeX entries, look up the arXiv number
on [INSPIRE-HEP](https://inspirehep.net). Which module does what is explained in
[The physics](physics.md).

## How to cite js-contrib

js-contrib has no DOI and no paper of its own. Cite:

1. **the repository:** <https://github.com/jhputschke/js-contrib>;
2. **the JETSCAPE framework paper** ([arXiv:1903.07706](https://arxiv.org/abs/1903.07706)),
   always;
3. **the papers of the modules you used**, from the table below and the lists on this page.

```bibtex
@misc{js-contrib,
  author       = {Putschke, Joern and others},
  title        = {{js-contrib}: contributions for the {JETSCAPE}/{X-SCAPE} framework},
  howpublished = {\url{https://github.com/jhputschke/js-contrib}},
  year         = {2026}
}
```

Add the version you used (a release tag or commit, and the container tag or the Mac wheel
version) in the text or a footnote. Every production file records the XML it was made with,
which is the precise record of the physics settings.

### Which papers for which production

| you use | cite |
|---|---|
| any js-contrib production | the [JETSCAPE framework](#framework); [3D MC-Glauber](#initial-state); [MUSIC](#hydrodynamics-and-equation-of-state) and the [HotQCD EoS](#hydrodynamics-and-equation-of-state); [Cornelius, Cooper–Frye and iSS](#freeze-out-and-particlization) |
| Au+Au, Pb+Pb or Z+jet with jets | in addition: [PYTHIA 8](#hard-process); [MATTER, LBT and the multistage approach](#jet-energy-loss); the [CausalLiquefier](#energymomentum-deposition) |
| O+O with ISR | in addition: the [X-SCAPE small-system papers](#framework); [PYTHIA 8](#hard-process); [MATTER and LBT](#jet-energy-loss) |
| FNO hydro (FnoHydro, PyFNOHydro) | in addition: [FNO and the FNO hydro paper](#machine-learning) |
| jets reconstructed with FastJet | [FastJet and anti-\(k_t\)](#jet-reconstruction) |
| SMASH run on the stored primary hadrons | [SMASH](#hadronic-transport) |

## Framework

- The JETSCAPE Collaboration (J. H. Putschke *et al.*), *The JETSCAPE framework*,
  [arXiv:1903.07706](https://arxiv.org/abs/1903.07706).
- The JETSCAPE Collaboration (A. Majumder *et al.*), *A multistage framework for studying the
  evolution of jets and high-\(p_T\) probes in small collision systems*, PoS (Hard Probes 2023),
  [arXiv:2308.02650](https://arxiv.org/abs/2308.02650).
- I. Soudi, W. Zhao, A. Majumder, C. Shen, J. H. Putschke *et al.*, *A soft-hard framework
  with exact four momentum conservation for small systems*,
  [arXiv:2407.17443](https://arxiv.org/abs/2407.17443).

Code: [JETSCAPE](https://github.com/JETSCAPE/JETSCAPE),
[X-SCAPE](https://github.com/JETSCAPE/X-SCAPE) (branch `contrib`), [jetscape.org](https://jetscape.org).

## Initial state

- C. Shen and B. Schenke, *Dynamical initial state model for relativistic heavy-ion
  collisions*, Phys. Rev. C 97, 024907 (2018),
  [arXiv:1710.00881](https://arxiv.org/abs/1710.00881).
- C. Shen and B. Schenke, *Longitudinal dynamics and particle production in relativistic
  nuclear collisions*, Phys. Rev. C 105, 064905 (2022),
  [arXiv:2203.04685](https://arxiv.org/abs/2203.04685).

Code: [3dMCGlauber](https://github.com/jhputschke/3dMCGlauber) (the version X-SCAPE fetches).

## Hard process

- T. Sjöstrand *et al.*, *An introduction to PYTHIA 8.2*, Comput. Phys. Commun. 191, 159
  (2015), [arXiv:1410.3012](https://arxiv.org/abs/1410.3012).
- C. Bierlich *et al.*, *A comprehensive guide to the physics and usage of PYTHIA 8.3*,
  SciPost Phys. Codebases 8 (2022), [arXiv:2203.11601](https://arxiv.org/abs/2203.11601).

Cite the paper of the PYTHIA version in your build. PYTHIA also does the string fragmentation
of the jet hadronization.

## Jet energy loss

- A. Majumder, *Incorporating space-time within medium-modified jet event generators*,
  Phys. Rev. C 88, 014909 (2013), [arXiv:1301.5323](https://arxiv.org/abs/1301.5323)
  (MATTER).
- S. Cao and A. Majumder, *Nuclear modification of leading hadrons and jets within a
  virtuality ordered parton shower*, Phys. Rev. C 101, 024903 (2020),
  [arXiv:1712.10055](https://arxiv.org/abs/1712.10055) (MATTER).
- Y. He, T. Luo, X.-N. Wang and Y. Zhu, *Linear Boltzmann transport for jet propagation in the
  quark-gluon plasma: elastic processes and medium recoil*, Phys. Rev. C 91, 054908 (2015),
  [arXiv:1503.03313](https://arxiv.org/abs/1503.03313) (LBT).
- T. Luo, S. Cao, Y. He and X.-N. Wang, *Multiple jets and \(\gamma\)-jet correlation in
  high-energy heavy-ion collisions*, Phys. Lett. B (2018),
  [doi:10.1016/j.physletb.2018.06.025](https://doi.org/10.1016/j.physletb.2018.06.025),
  [arXiv:1803.06785](https://arxiv.org/abs/1803.06785) (LBT).
- The JETSCAPE Collaboration (S. Cao *et al.*), *Multistage Monte-Carlo simulation of jet
  modification in a static medium*, Phys. Rev. C 96, 024909 (2017),
  [arXiv:1705.00050](https://arxiv.org/abs/1705.00050).
- The JETSCAPE Collaboration (A. Kumar *et al.*), *Inclusive jet and hadron suppression in a
  multistage approach*, Phys. Rev. C 107, 034911 (2023),
  [arXiv:2204.01163](https://arxiv.org/abs/2204.01163) (MATTER + LBT).

## Energy–momentum deposition

- Y. Tachibana, C. Shen and A. Majumder, *Bulk medium evolution has considerable effects on
  jet observables*, Phys. Rev. C 106, L021902 (2022),
  [arXiv:2001.08321](https://arxiv.org/abs/2001.08321) (the causal liquefier).

## Hydrodynamics and equation of state

- B. Schenke, S. Jeon and C. Gale, *3+1D hydrodynamic simulation of relativistic heavy-ion
  collisions*, Phys. Rev. C 82, 014903 (2010),
  [arXiv:1004.1408](https://arxiv.org/abs/1004.1408) (MUSIC).
- B. Schenke, S. Jeon and C. Gale, *Elliptic and triangular flow in event-by-event (3+1)D
  viscous hydrodynamics*, Phys. Rev. Lett. 106, 042301 (2011),
  [arXiv:1009.3244](https://arxiv.org/abs/1009.3244) (MUSIC).
- B. Schenke, S. Jeon and C. Gale, *Higher flow harmonics from (3+1)D event-by-event viscous
  hydrodynamics*, Phys. Rev. C 85, 024901 (2012),
  [arXiv:1109.6289](https://arxiv.org/abs/1109.6289) (MUSIC's Kurganov–Tadmor scheme).
- J.-F. Paquet, C. Shen, G. S. Denicol, M. Luzum, B. Schenke, S. Jeon and C. Gale,
  *Production of photons in relativistic heavy-ion collisions*, Phys. Rev. C 93, 044906
  (2016), [arXiv:1509.06738](https://arxiv.org/abs/1509.06738) (MUSIC with shear and bulk
  viscosity).
- A. Bazavov *et al.* (HotQCD), *The equation of state in (2+1)-flavor QCD*, Phys. Rev. D 90,
  094503 (2014), [arXiv:1407.6387](https://arxiv.org/abs/1407.6387).
- The JETSCAPE Collaboration (D. Everett *et al.*), *Multisystem Bayesian constraints on the
  transport coefficients of QCD matter*, Phys. Rev. C 103, 054904 (2021),
  [arXiv:2011.01430](https://arxiv.org/abs/2011.01430) (the \(\eta/s(T)\) and \(\zeta/s(T)\)
  parametrizations).

**The GPU port.** MUSIC4GPU, the CUDA and Metal port of MUSIC that the productions run, has no
publication of its own; it ports MUSIC's CPU algorithms without changing the physics. Cite the
MUSIC papers above and the repository. Code: [MUSIC](https://github.com/MUSIC-fluid/MUSIC),
[MUSIC4GPU](https://github.com/jhputschke/MUSIC4GPU) (branch `XSCAPE`).

## Freeze-out and particlization

- P. Huovinen and H. Petersen, *Particlization in hybrid models*, Eur. Phys. J. A 48, 171
  (2012), [arXiv:1206.3371](https://arxiv.org/abs/1206.3371) (Cornelius).
- F. Cooper and G. Frye, *Single-particle distribution in the hydrodynamic and statistical
  thermodynamic models of multiparticle production*, Phys. Rev. D 10, 186 (1974),
  [doi:10.1103/PhysRevD.10.186](https://doi.org/10.1103/PhysRevD.10.186).
- C. Shen, Z. Qiu, H. Song, J. Bernhard, S. Bass and U. Heinz, *The iEBE-VISHNU code package
  for relativistic heavy-ion collisions*, Comput. Phys. Commun. 199, 61 (2016),
  [arXiv:1409.8164](https://arxiv.org/abs/1409.8164) (iSS).

Code: [iSS](https://github.com/jhputschke/iSS) (branch `smash_ready`, the version X-SCAPE
fetches).

## Hadronic transport

- J. Weil *et al.*, *Particle production and equilibrium properties within a new hadron
  transport approach for heavy-ion collisions*, Phys. Rev. C 94, 054905 (2016),
  [arXiv:1606.06642](https://arxiv.org/abs/1606.06642) (SMASH).

The productions use SMASH's hadron list but do not run SMASH; see
[The physics](physics.md#11-hadronic-afterburner).

## Machine learning

- Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart and
  A. Anandkumar, *Fourier neural operator for parametric partial differential equations*,
  ICLR 2021, [arXiv:2010.08895](https://arxiv.org/abs/2010.08895).
- D. Stewart and J. Putschke, *Fast prediction of the hydrodynamic QGP evolution in
  ultrarelativistic heavy-ion collisions using Fourier neural operators*, Phys. Rev. C 113,
  014904 (2026), [arXiv:2507.23598](https://arxiv.org/abs/2507.23598). Data and trained
  models: [doi:10.5281/zenodo.16647726](https://doi.org/10.5281/zenodo.16647726).

```bibtex
@article{Stewart:FNOhydro,
  author  = {Stewart, David and Putschke, Joern},
  title   = {Fast prediction of the hydrodynamic {QGP} evolution in ultrarelativistic
             heavy-ion collisions using {F}ourier neural operators},
  journal = {Phys. Rev. C},
  volume  = {113},
  number  = {1},
  pages   = {014904},
  year    = {2026},
  doi     = {10.1103/jsh8-6tm7},
  eprint  = {2507.23598},
  archivePrefix = {arXiv}
}
```

The 3+1D FNO work that the background/jet pair productions are made for has no publication
yet.

## Jet reconstruction

- M. Cacciari, G. P. Salam and G. Soyez, *FastJet user manual*, Eur. Phys. J. C 72, 1896
  (2012), [arXiv:1111.6097](https://arxiv.org/abs/1111.6097).
- M. Cacciari, G. P. Salam and G. Soyez, *The anti-\(k_t\) jet clustering algorithm*, JHEP 04
  (2008) 063, [arXiv:0802.1189](https://arxiv.org/abs/0802.1189).

## Measurements the Z+jet setup is built for

- CMS Collaboration, *Study of jet quenching with Z+jet correlations in PbPb and pp collisions
  at \(\sqrt{s_{NN}}\) = 5.02 TeV*, Phys. Rev. Lett. 119, 082301 (2017),
  [arXiv:1702.01060](https://arxiv.org/abs/1702.01060).
- CMS Collaboration, *Using Z boson events to study parton-medium interactions in PbPb
  collisions*, Phys. Rev. Lett. 128, 122301 (2022),
  [arXiv:2103.04377](https://arxiv.org/abs/2103.04377).
- CMS Collaboration, *Evidence of medium response to hard probes using correlations of Z bosons
  with hadrons in heavy ion collisions*, Phys. Lett. B 874, 140120 (2026),
  [arXiv:2507.09307](https://arxiv.org/abs/2507.09307).
- ATLAS Collaboration, *Medium-induced modification of Z-tagged charged particle yields in
  Pb+Pb collisions at 5.02 TeV with the ATLAS detector*, Phys. Rev. Lett. 126, 072301 (2021),
  [arXiv:2008.09811](https://arxiv.org/abs/2008.09811).

## Next steps

- [The physics](physics.md): where each of these models enters.
- [Contribs and repositories](../reference/contribs.md): all code repositories involved.
