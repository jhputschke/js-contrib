# hard_sub_compare — the effect of the hard-momentum subtraction in O+O

A quick estimate of what 3dMCGlauber's `Subtract_hard_momentum` does with ISR: two campaigns
of [`../run_prod_oo_isr.py`](../run_prod_oo_isr.py) that differ only in that setting, and a
notebook that compares their hadrons.

Both campaigns ran with Matter alone, the O+O energy loss before 2026-10-21; `run_compare.sh`
passes `--matter-only` so that jobs added to them stay so. Their jet hadrons were made before
the colour fixes of X-SCAPE `lbt_colored_hadro` (duplicate colour tags in most events), and in
`sub/` seeds 0103 (from event 46) and 0129 (from event 3) Pythia's failed fragmentation wrote
the partons out as jet hadrons: leave those two jobs out of the jet-hadron comparisons, or
rerun them ([`../README.md`](../README.md), "Jet hadronization colour checks").

| file | purpose |
|---|---|
| `run_compare.sh` | runs both campaigns (resumable) |
| `hard_sub_compare.ipynb` | the comparison: the centrality bias (multiplicity vs Npart, centrality classes from the multiplicity), bulk and jet-hadron dN/dη, dE/dη and pT spectra, the energy as a function of the parton pT, a table per pTHat window; Ncoll-weighted |
| `sub/`, `nosub/` | the campaigns' files (not in git, see `.gitignore`) |

| campaign | folder | option |
|---|---|---|
| subtracted (the default) | `sub/` | — |
| not subtracted (reference) | `nosub/` | `--no-subtraction` |

Both: O+O 200 GeV, 0–80% centrality in the geometry (b < 5.6 fm, see below),
`--pthat-bins 10-20,20-30,30-40 --parton-ymax 0.6 --parton-y-mode both` (both outgoing
hard partons at |y| < 0.6, before the ISR), hadrons only
(`--bulk-hadrons 10 --no-particlize --no-hydro-file --no-x`). 40 jobs × 75 events, seeds
101–140 in both, so 1000 events per window and campaign, and the same Pythia hard events in
the two (3dMCGlauber's geometry differs from a job's second event on: compare ensembles).

```bash
conda activate js_fno
./run_compare.sh > run_compare.log 2>&1 &       # ~1.3-1.4 h per campaign on the GB10 (-j 8, MPS)
jupyter nbconvert --to notebook --execute --inplace hard_sub_compare.ipynb
```

`NJOBS`, `EVENTS`, `SEED0`, `P`, `B_MIN`, `B_MAX`, `BLOCK` and `OMP_NUM_THREADS`
(environment) change the defaults. The two campaigns alternate in blocks of `BLOCK` (16)
jobs, so a stopped run leaves about equal statistics in both.

**Centrality.** No centrality cut beyond the 80% experiments can measure, so that a bias of
the subtraction shows up rather than being cut away. b < 5.6 fm is the 80th percentile of
b of inelastic O+O at 200 GeV with this XML's 3dMCGlauber settings: 100k events of the
standalone 3dMCGlauber (`only_event_statistics 1`, b < 12 fm, Npart ≥ 2) give σ_inel =
1.55 b and the percentiles 10% 1.91, 20% 2.71, 40% 3.84, 60% 4.71, 80% 5.59 fm (not the
geometric √(cσ/π): the collision probability is well below 1 at large b). In 0–80%,
⟨Npart⟩ = 11.3 and 7.5% of the events have one collision only (18% in 0–100%); the
most peripheral 20% hold 3.5% of the hard events (Ncoll-weighted). `B_MAX=12` runs 0–100%.

**Events without a medium.** When the only collision is the hard one, the hard pair makes
no string and there is no medium: the bulk unit is empty and Matter showers in vacuum.
This needs X-SCAPE's fix on branch `oo_hard_sub_compare` (`EvolutionHistory::get` returns
the vacuum cell for an empty history); before it, such an event after one with a medium
stopped the job.

The notebook reads only finished jobs (those with a `.json`), so it can also run while the
campaigns are going; `HSC_DIR` points it at another folder with `sub/` and `nosub/`.

The per-event record the notebook uses (pTHat, window, Npart, `isr_sub_*`, `remnant_*`,
`strings_*`, the hard partons' y and pT) is in the hadron files' `units/`, since these jobs
write no hydro or particlize file.
