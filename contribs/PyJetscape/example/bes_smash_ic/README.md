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

Production settings: no `--checks`, one iSS sample, SMASH's tabulations already
in `SMASH_TABULATIONS`. A second event in the same job takes about 50 s: SMASH
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
