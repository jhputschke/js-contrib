<!-- Plan, written 2026-10-01. Status: not started. -->

# Plan: particle decays at the analysis level (π⁰ and weak decays of the hadron files)

## Context

The hadron files of a production (`hadronize.py` → `<stem>_hadrons_{bulk_bg,bulk_jet,jet_frag}.h5`,
`run_h5toROOT.py` → `<stem>_hadrons.root`) hold the hadrons after iSS's own decays. iSS decays
every particle whose first decay channel in its table has more than one product
(`iSS/src/particle_decay.cpp:83`). With MUSIC's EOS 9 that table is `pdg-urqmd_v3.3+.dat`:

- **decayed:** the strong resonances (ρ, ω, K\*, φ, Δ, N\*, Σ\*, Λ\*, Ξ\*, …, 42 mesons and
  117 baryons up to 2.25 GeV) and the electromagnetic decays the table lists (η → γγ / 3π,
  Σ⁰ → Λγ, ω → π⁰γ, η′ → ργ / ωγ), down to stable particles;
- **stable (16, with antiparticles):** γ, π±, **π⁰**, K±, **K⁰/K̄⁰**, p, n, **Λ, Σ±, Ξ⁰, Ξ⁻, Ω⁻**.

So π⁰ is never decayed, and neither are the weak decays. The jet production checked this:
job 0001 of `AuAu_0_10_pth10-40_eta06_c1` has exactly these 13 |pid|s, and no resonance.
The photons in the files come from η, ω and Σ⁰ only.

Charged-hadron analyses don't need anything more. Photons, neutral energy, π⁰ → γγ pairs, and a
final state with weak-decay products all need a decayer after iSS. It belongs to the analysis
and not to iSS: the hadron files stay the physics record (primary hadrons after strong and EM
decays), and every choice below can be made per analysis without hadronizing again.

### Why not the alternatives

| option | problem |
|---|---|
| change iSS's table (`111 → 22 22`) | changes every iSS run that reads the table, the jet production's hadrons too; it needs a new hadronization of every campaign, and the choice isn't per analysis |
| Pythia 8 as the decayer | complete tables and matrix elements, but a Python loop over ~6.5 M hadrons per jet-leg file; a heavy dependency for a dozen 2-body decays |
| decayed copies of the `.h5` files | simplest to read, and cheap (the hadron files of the 67-file campaign are 4.6 GB), but one copy per choice of decays, kept in sync with the originals by hand; the readers would still need the `mother` field |

## Decay modes

Every decay below is 2-body, except the π⁰ Dalitz decay. One vectorized 2-body routine,
isotropic in the rest frame, covers them. Isotropic ignores Λ, Ξ, Ω polarization and decay
asymmetries, which is fine at this level.

| mode | decays | adds (job 0001 jet leg: 6.5 M hadrons, 1.65 M π⁰) |
|---|---|---|
| `pi0` | π⁰ → γγ | +1.65 M γ (+25%) |
| `weak` | `pi0`, plus K⁰/K̄⁰ → 50% K⁰_S, 50% K⁰_L; K⁰_S → π⁺π⁻ / π⁰π⁰; Λ → pπ⁻ / nπ⁰; Σ⁺ → pπ⁰ / nπ⁺; Σ⁻ → nπ⁻; Ξ⁰ → Λπ⁰; Ξ⁻ → Λπ⁻; Ω⁻ → ΛK⁻ / Ξ⁰π⁻ / Ξ⁻π⁰ (and their products, down to stable) | about +0.5 M from the weak decays (+8%), plus their π⁰ → γγ |
| a set of pids | only those, e.g. `{111, 310}` | — |

Stable in every mode: γ, e±, μ±, π±, K±, K⁰_L, p, n. K⁰_L, K± and π± have cτ of metres.

**Which mode for what.** `pi0` is the "primary" final state of the usual experimental
definition (ALICE: a primary particle has cτ > 1 cm; K⁰_S at 2.7 cm and Λ at 7.9 cm are
primaries). `weak` is the final state after every weak strange decay. It gives the
feed-down a detector sees when it doesn't reject secondaries. Neither models a detector.

**π⁰ Dalitz** (e⁺e⁻γ, 1.17%): phase 1 decays π⁰ to γγ only, with branching ratio 1. A
3-body option with the Kroll–Wada distribution can come later if e± matter.

**Constants.** Masses, branching ratios and cτ from the PDG (2024 tables), in one small table
in the module, with the source written next to each value. Not read from the iSS file: its
UrQMD list has no K⁰_S/K⁰_L, and its masses are rounded (π at 0.138).

## Where it sits

```
hadron .h5 files ──► jetscape.hadrons_h5.HadronFileReader(..., decays="pi0")  ──► Python analyses
          │                       uses jetscape.decays
          └──► run_h5toROOT.py --decays pi0 ──► <stem>_hadrons.root ──► HadronFileReader.h, macros
```

- **`jetscape/decays.py`** (new): `Decayer(mode)`, and `Decayer.decay(sample, key)` on the
  per-sample dict the readers return (`pid`, `pstat`, `p`, `x`). It returns the same dict with
  the daughters appended, plus `mother` (int32: −1 for an iSS hadron, else the row of its
  mother) and the decayed parents removed (`keep_decayed=True` keeps them, `pstat` = 12).
  Daughters get `pstat` = 13. iSS's hadrons keep 11.
- **`hadrons_h5.HadronFileReader(..., decays=None)`**: every sample it returns
  (`sample_event`, `jet_event`, the background) goes through the decayer. With `None`,
  nothing changes, bit for bit.
- **`run_h5toROOT.py --decays MODE`**: writes the decayed hadrons and a `mother` column, and
  records the mode as a file attribute (`decays`). `HadronFileReader.h` reads `mother` when the
  column is there. The C++ side needs no decayer.
- **`wake_hadrons.py --decays MODE`**, and the notebooks' settings, pass it on.

**Positions** (`x` = t, x, y, z, when stored). A daughter gets its mother's freeze-out point.
Flight paths aren't modelled: π⁰'s is 25 nm; the weak decays' are centimetres, outside the
fm-scale picture the positions describe. The `mother` index says which hadrons are decay
products.

## Random numbers

- **Reproducible and addressable.** No stream: each decay draws from a counter-based hash
  (64-bit, e.g. SplitMix64 over the key words, vectorized in numpy). The key is (mode, sample
  index k, the hadron's pid, the float32 bits of its four-momentum, decay step). The same file
  and mode give the same decays everywhere: in Python, in the ROOT export, on the fly or not.
  The order of the hadrons doesn't matter.
- **Correlated oversamples.** With `hadronize.py --correlated`, sample k of the background and
  of the jet leg are pairs: iSS addresses its random numbers by (species, block of cells,
  hadron index) and decays each hadron from a stream of its own
  ([`PLAN_iSS_optim.md`](PLAN_iSS_optim.md), Part B), so where the jet changed nothing the two
  legs have the same hadrons, bit for bit. Keying on the hadron itself makes those
  identical hadrons decay identically in both legs. The wake (jet leg − background) then
  keeps the cancellation correlated sampling gives it. A key on the row index would not: the
  rows of the two legs don't line up.
- **Collisions.** Two identical hadrons (same pid and float32 momentum) in one sample would
  decay the same way. That doesn't happen in practice, and it would be harmless.

## Phases

| phase | content | effort |
|---|---|---|
| 0 | constants table (PDG 2024), the key/hash, the 2-body kinematics (rest-frame decay, boost), unit tests | ½ day |
| 1 | mode `pi0` in `jetscape.decays`; `HadronFileReader(decays=)`; tests | ½ day |
| 2 | mode `weak` and pid sets: chains (Ω → Ξ → Λ → pπ), K⁰ → K⁰_S/K⁰_L, `mother` index | 1–2 days |
| 3 | `run_h5toROOT.py --decays`, `mother` in the ROOT layout and `HadronFileReader.h`, `wake_hadrons.py --decays` | 1 day |
| 4 | validation (below), docs (`analysis/README.md`, `root_export/README.md`, `analysis_root/README.md`) | ½–1 day |

## Validation

1. **Conservation.** Every decay conserves four-momentum to float32 precision, and each
   daughter is on its mass shell.
2. **Branching ratios** recovered within statistics, per channel.
3. **Invariant masses.** γγ pairs from one π⁰ peak at m(π⁰) with zero width; pπ⁻ from Λ at
   m(Λ); π⁺π⁻ from K⁰_S at m(K⁰_S).
4. **Isotropy.** The daughters' angle in the mother's rest frame is flat in cos θ.
5. **Yields.** On job 0001: `pi0` adds 2 γ per π⁰ and removes the π⁰; `weak` raises the
   charged-hadron count by what the branching ratios predict (e.g. 63.9% of Λ give a p and a
   π⁻).
6. **Bit-identical without decays.** `decays=None` and no `--decays` give today's results,
   checked against the stored outputs of `analysis/hadron_distributions.ipynb` and
   `analysis_root/read_hadrons.ipynb`.
7. **Python vs ROOT.** `HadronFileReader(decays=m)` and `run_h5toROOT.py --decays m` +
   `HadronFileReader.h` give the same hadrons, in the same way `tests/test_analysis_root_reader.py`
   compares them today.
8. **Correlated pairs.** On a `--correlated` file: identical hadrons of the two legs' sample k
   have identical daughters, and the wake's error with `pi0` is close to the one without.
9. **Reference (optional).** Pythia 8's decays of the same π⁰, K⁰_S and Λ give the same
   distributions.

## Open points

- `keep_decayed` default: drop the parents (a final state) or keep them, flagged (a history)?
  Proposed: drop, as a detector would.
- Charged-hadron selections (`CHARGED` in `hadrons_h5`, `set_charged_eta` in the C++ reader):
  with `weak`, the p and π± from decays count as charged hadrons. Whether an analysis wants
  that is the analysis's choice; the `mother` index allows both.
- Speed: ~1.65 M π⁰ per jet-leg file is a few vectorized operations, well under a second.
  Not measured yet.
