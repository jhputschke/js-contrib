# bes_smash_ic: SMASH → MUSIC → iSS → SMASH at beam-energy-scan energies

X-SCAPE's `SMASHInitialCondition` chain with net baryon number, electric charge
and strangeness carried through the hydro, on the GPU:

1. **SMASH** runs the collision until every hadron has crossed the hyperbola
   τ = 0.5 fm (`<IC_particle_extraction_tau>`).
2. **MUSIC4GPU** (CUDA) evolves the fluid the hadrons are deposited into
   (`HadronicLiquefier`, covariant kernel), with the NEOS-4D equation of state
   (EOS 20), ρ_B, ρ_Q, ρ_S and their source terms, shear and bulk viscosity with
   μ_B-dependent η/s and ζ/s, and optionally baryon diffusion. The freeze-out
   surface of every step goes to
3. **iSS**, which samples hadrons with the cell's B, Q, S densities;
4. **SMASH** runs the afterburner.

This is a C++ executable of X-SCAPE, not a PyJetscape driver. Stages 1–3 run
one after the other ("staged"); see the
[Cartesian MUSIC on GPU](https://claude.ai/artifact/5wsPQCiiPdafvyHpDcgpUi)
note for the concurrent (Cartesian) alternative.

## Requirements

- An X-SCAPE CUDA build with SMASH: `-DUSE_CUDA=ON -DUSE_MUSIC=ON -DUSE_ISS=ON
  -DUSE_SMASH=ON`, after `external_packages/get_smash.sh` (SMASH-3.2.2). SMASH-3.2.2
  requires Pythia **8.315 exactly**; if X-SCAPE itself links another Pythia
  (the conda env), build 8.315 separately, point SMASH's CMake at it
  (`-DPythia_CONFIG_EXECUTABLE`, `-DPythia_LIBRARY`, `-DPythia_INCLUDE_DIR`),
  configure X-SCAPE with `-DPYTHIA8_INCLUDE_DIR`/`-DPYTHIA8_LIBRARY` of the same
  install and pass `--pythia <prefix>` here (the two must not be mixed in one
  process).
- MUSIC4GPU with the B, Q, S port (branch `bqs_gpu`, see its `docs/BQS_GPU.md`).
- The NEOS-4D tables: `cd external_packages/music4gpu/EOS && bash
  download_Neos4D.sh` (UrQMD variant), then re-run CMake so the build tree's
  `EOS/` has them. iSS's `HRG4D` tables come with `get_iSS.sh`.

## Run

```bash
./run_bes_smash_ic.sh --build ../../../../../../build_gpu --seed 1 OUT
./run_bes_smash_ic.sh --seed 1 --checks OUT              # + per-step B, Q, S and the charge check
./run_bes_smash_ic.sh --seed 1 --checks --cpu OUT        # the same event with MUSIC on the CPU
./run_bes_smash_ic.sh --seed 1 --checks --samples 100 OUT  # 100 iSS samples per surface
```

`--sqrts`, `--b MIN MAX`, `--events`, `--diffusion`: see `--help`. Each job runs
in its own directory `OUT/<tag>` and can run next to others.
`bqs_balance.py run.log hadrons.dat` prints the charge balance of a `--checks`
run: SMASH → sources → grid → surface → iSS samples → final hadrons.

## Validation and timing (GB10, Au+Au, √s_NN = 19.6 GeV, b = 6–8 fm, seed 42)

Grid 150 × 150 × 140 (x, y ±15 fm, η_s ±7), Δτ = 0.02 fm, about 330 steps to
freeze-out. 67 baryons, net charge 26 deposited.

| run | MUSIC | wall time | grid N_B, N_Q at the end |
|---|---|---|---|
| A0, before the port | CPU (20 threads), EOS 20 | 23 min 15 s | 65.50, 25.03 |
| GPU port, `--checks` | GPU, EOS 20 | 19 min 43 s | 65.79, 25.14 |
| GPU port, production settings | GPU, EOS 20 | 17 min 43 s | |
| now, `--checks --samples 100` | GPU, EOS 20 | 1 min 29 s | 65.65, 25.03 |
| now, production settings | GPU, EOS 20 | **1 min 01 s** | |
| now, production settings, local arm64 image | GPU, EOS 20 | 0 min 59 s | |
| now, production settings, `--cpu` | CPU (16 threads), EOS 20 | 16 min 00 s | |

Production settings: no `--checks`, one iSS sample, SMASH's tabulations already
in `SMASH_TABULATIONS`. The image (CUDA 13.2.1) and the host build (CUDA 13.0)
agree within float differences (8 of 580,826 surface cells), so their sampled
hadrons differ; each is reproducible run to run. A second event in the same job takes about 50 s: SMASH
builds its decay-width tables (about 12 s) once per process, so run several
events per job (`--events`).

Where the time went: the hadronic liquefier's sources were asked at every cell
of the 150 × 150 × 140 grid in every step (now per droplet, only in the steps it
deposits), iSS re-read its EOS tables for every surface, MUSIC copied the grid
to the host after every clock tick and for every freeze-out check, and SMASH
re-tabulated its integrals at every start. The B, Q, S balance (100 iSS samples):
through the surface 65.00, 24.94, −0.14; in the samples 64.61, 25.05, −2.2
(sources 67, 26, 0).

The step-by-step GPU hydro now also starts each clock tick from the real
previous step, and its viscous regulator treats fast, dilute cells as the CPU
does; the final state is about twice as close to the CPU's (MUSIC4GPU
`docs/BQS_GPU.md`).

## Dynamic fluidization (`--dynamic`, X-SCAPE branch `trackB`)

```bash
./run_bes_smash_ic.sh --sqrts 7.7 --b 0 3 --seed 1 --dynamic OUT
```

SMASH runs the whole event (to 40 fm/c) next to a Cartesian MUSIC that starts
at t = 0 (x, y, z ±12 fm in 0.4 fm cells, Δt = 0.02 fm/c). Every clock tick
(0.1 fm/c) the hadrons that pass the fluidization criterion of
[arXiv:2507.19389](https://arxiv.org/abs/2507.19389) leave SMASH and become
hydro sources (rest-frame energy density of the covariant Gaussian sum, without
the hadron's own, at least 0.5 GeV/fm³; string fragments only 0.4 fm/c after
their production; never unscattered nucleons, nor hadrons that came from the
fluid), and the hadrons iSS samples from the tick's freeze-out surface enter
SMASH at their own time. MUSIC4GPU runs Cartesian coordinates on the GPU (CUDA).
The job prints the exchange totals, the time per module and the final state
against the nuclei (`conservation.py`).

| Au+Au 7.7 GeV, b = 0–3 fm, seed 1, GB10 | wall time |
|---|---|
| hydro on the CPU (B3) | 5 min 14 s |
| hydro on the GPU, first event of a job | 30 s |
| hydro on the GPU, further events of the job | about 15 s |

The first event pays SMASH's Pythia set-up (one Pythia object per hadron pair of
its hard strings, about 14 s, once per job): run several events per job. In
the further events MUSIC takes about 10 s (840 steps of 60³ cells), iSS 3.4 s
(about 160 surfaces), SMASH about 1 s, the fluidization criterion 0.4 s.

**Not yet conserved.** In 5 events E came out 103–108 % of the nuclei's and
B 400–421 of 394. The excess comes back through particlization: in seed 1 the
fluidized hadrons carry B = 299, the particlized ones B = 326, and the final
state has 27 baryons too many. Cooper–Frye counts only the outgoing part of
each surface piece (p·dσ > 0); on the surface of a fluid that keeps forming
from sources, many pieces have spatial normals and inflow, which is dropped.
