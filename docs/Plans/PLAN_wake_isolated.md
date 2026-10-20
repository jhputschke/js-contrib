# PLAN: the Z+jet wake on its own, in (Δη, Δφ), without a mixed event

2026-10-20. Why: the liquefier source scan (`analysis/zjet/wake_scan/`, `tau_delay` 0.5–2 fm,
`e_threshold` 1–3 GeV) left CMS 2507.09307's Z-side Δy of Pb+Pb − pp unchanged within its
errors. That observable is not sensitive to the wake's shape: integrated over Δy, each piece's
CMS value is half its Z-side minus away-side yield, i.e. the odd Δφ harmonics, mostly the
dipole, and the wake's dipole is largely fixed by the momentum the jet deposits, whatever the
source looks like. The scan saw this as a Z-side share of the wake that hardly changed between
values.
In the model the wake can be taken on its own instead: the paired ⟨bulk_jet⟩ − ⟨bulk_bg⟩
(each event's jet leg against its own background, correlated samples, sample k against sample
k) is the wake hadrons and nothing else, and `jet_frag` is the fragments. Both are per-Z truth
yields: no mixed event, no mean subtraction, no normalization beyond "per Z". This plan
measures them in (Δη, Δφ), around the Z and around the recoiling parton, for every value of the
scan, and reads off what the source does to the wake's shape.

> **Status 2026-10-20** (js-contrib branch `wake_iso` off main 0df9477): step 1 implemented
> (`analysis/zjet/wake_iso.py`, `analysis/zjet/wake_scan/compare_iso.py`, `make_data.sh iso`,
> `tests/test_wake_iso.py`), checked (`wake_iso.py check`: the Z frame = `zjet_wake`'s to
> 10⁻¹⁵, Pb+Pb and pp) and run on the scan's hadron files on the GB10; results in
> `runs/wake_scan/WAKE_ISO_SUMMARY.md` (git-ignored: not public). Steps 2 and 3 not started.

**Decisions (user, 2026-10-20):** step 1 first, on the existing scan files, with the Z frame and
the recoil-parton frame; no extra job to split the droplets by shower (the share of the wake
that comes from the MPI/ISR partons stays unmeasured in step 1).

## 0. What "the wake on its own" contains

- **Everything the deposition adds or removes.** The thermal bulk cancels exactly in the mean
  (the two legs are the same fluid without and with the source); correlated sampling only
  makes the sampling noise small. Nothing is subtracted beyond that.
- **The wakes of all the partons the energy loss gets**, not only the Z's recoil. With MPI and
  ISR on (the campaign XML), the scan's events carry ~14 parton initiators each; besides the
  recoil (~52 GeV p_T) they carry ~43 GeV of p_T inside |y| < 2.4, with a net −4 GeV along the
  Z (measured on `tau_delay_1p0`, 192 events). Their directions are random relative to the Z,
  so on average their wake hadrons are flat in Δφ around the Z: real wake hadrons, not the
  recoil's. Only a source restricted to the recoil (step 3) separates them.
- **The recoil's own broad response**: deposited energy that thermalizes and flows out with the
  bulk, all around, also on the Z side (the scan's wake has a positive Z-side yield).

So the maps are read through quantities a part flat in Δφ does not enter: the Δφ harmonics
n ≥ 1, and differences between Δφ regions at the same Δη. Nothing is fitted or subtracted.

The Z frame smears the wake's Δη shape: the recoil's rapidity relative to the Z has an RMS of 1.2
(52 % within CMS's first bin, 0.8). The recoil-parton frame shows the wake's own Δη extent.

## 1. Step 1: the maps from the scan's files (no new jobs)

**Inputs.** `runs/wake_scan/data/<setting>_<value>/` (five runs: `tau_delay` 0.5, 1.0, 2.0;
`e_threshold` 1.0, 3.0; `tau_delay_1p0` is the baseline of both), four campaign-format jobs each
(seeds 102–105, 48 events, two hydro backgrounds each, 40 oversamples, 40 fragmentations,
correlated sampling); the pp reference `runs/pp_comp/data/campaign/` (12,000 events,
fragments only).

**Selection and weights** as everywhere in the Z–hadron analysis: CMS 2507.09307's Z cuts
(`zjet_wake.CMS_2507`), every event weighted by its window's cross section over its events
× N_coll (`zjet_wake.event_weights`), tracks charged with |η| < 2.4, never the Z's leptons.

**Contents** per Z: `wake` = ⟨bulk_jet⟩ − ⟨bulk_bg⟩ (per-sample means, each event against its own
background), `frag` = ⟨jet_frag⟩; pp `frag` only; Pb+Pb − pp of the fragments.

**Frames.**

| frame | Δη | Δφ ∈ [−π/2, 3π/2) | Z's |
|---|---|---|---|
| `Z` | η − y_Z | φ − φ_Z (the recoil near π) | all selected |
| `ZJ` | η − y_Z | φ − φ_Z | those with \|η_rec\| < 1 |
| `J` | η − η_rec | φ − φ_rec (the Z near π) | those with \|η_rec\| < 1 |

The recoil is the hardest initiator that is a parton (quark or gluon), back-to-back with the Z
within π/2 in 99.5 % of the scan's events. The cut |η_rec| < 1 keeps |Δη| < 1.4 inside the
track acceptance in the `J` frame; `ZJ` is the same Z's in the Z frame, to see the smearing on
the same events.

**What is stored** (`wake_iso.py map`, one `.npz` per run): per track p_T class (0.5–1 GeV,
model only; CMS's 1–2, 2–4, 4–10 GeV) the map in (Δη 0.2, Δφ π/16), the exact harmonics
Σ cos(nΔφ), n = 0–4, per Δη bin, and the p_T spectra toward the Z and toward the recoil; per
event the totals of every stored hadron (|η| < 2.6, any species and p_T): N, E, p_T,
p_T cos Δφ_Z, p_T cos Δφ_rec, and the deposited droplet energy (`units/E_droplets`). All as
weighted sums per hydro background (pp: per block of events), so the per-Z means and the
jackknife over backgrounds follow at load time, and two runs with the same seeds pair
background by background.

**What is read off** (`wake_scan/compare_iso.py`), per run and as paired differences to the
baseline (the same events and backgrounds; jackknife over the same background blocks):

1. the maps (baseline) of the wake, the fragments and the fragments' Pb+Pb − pp, both frames;
2. projections: Δφ (all Δη) and Δη on the Z side in the Z frame; Δφ (|Δη| < 1.4) and Δη near
   the recoil (|Δφ| < π/2) in the `J` frame;
3. the harmonics c_n = Σ cos(nΔφ) per Z, n = 1–4, per p_T class (Z frame all Δη, `J` frame
   |Δη| < 1.4);
4. the local dip toward the Z: the density at |Δφ − Δφ_Z| < π/8 minus that at
   π/4 < |Δφ − Δφ_Z| < π/2, at |Δη| < 1 (both frames; a diffusion wake is a negative value);
5. the wake's Δη extent around the recoil: the near-side (|Δφ| < π/2) density at |Δη| < 0.4 and
   at 0.4 < |Δη| < 1.4;
6. the wake's p_T spectra toward the Z and toward the recoil;
7. the momentum: Σ p_T cos Δφ_Z of all wake hadrons in |η| < 2.6 (for `tau_delay` the droplets
   are identical, so the in-acceptance dipole momentum should not change, while the counts per
   p_T class do), and the wake's energy in |η| < 2.6 as a fraction of the deposited droplet
   energy (not a closure: the files stop at |η| = 2.6, and the MPI partons deposit everywhere).

**Checks.** `wake_iso.py check`: the `Z` frame wake and fragment maps of one file against
`zjet_wake.z_frame_yields` (the reader's paired difference), to rounding. `compare_iso.py`
refuses runs whose events, weights or backgrounds differ.

**Size and precision.** 108 Z's on 8 backgrounds (N_eff 90) per value: each value's maps and
projections are usable; value-to-value differences are significant in projections, harmonics
and totals, not bin by bin in 2D (the scan's wake changes are small against the per-bin
sampling noise). The jackknife has 8 backgrounds (7 degrees of freedom): errors good to ~25 %.

**Cost.** One pass over 20 jobs' hadron files (~4.7 GB) and the pp file: minutes on the GB10 with
one process per file.

## 2. Step 2: shared sampling seeds between the values of a scan

The scan's jobs seed iSS and Colorless from each job's random `file_uuid`, so two values of one
seed share partons, droplets and backgrounds but not the sampling: the paired differences carry
√2 × the wake's own sampling noise, and the fragments' differences for `tau_delay` (same
partons) are pure noise. A `run_prod_jet.py` option that keys the seeds by the job seed instead
(`hz.uuid_key(file_uuid)` in the `--hadrons-only` path) makes `bulk_bg` and, for `tau_delay`,
`jet_frag` bit-identical between values, and the difference jet leg A − jet leg B correlated:
at least 2× less variance. Then rerun the scan (~2 h per three values on the GB10) and step 1 on
it.

## 3. Step 3: the recoil's own wake

A variant in which MUSIC_2 receives only the droplets of the recoil's shower, paired like
`tau_delay` (same jets, same MUSIC_1): "all − recoil only" is the MPI/ISR partons' wake. Needs a
droplet filter by shower in X-SCAPE's liquefier (the droplets' shower assignment is known in
`wake_observables.assign_droplets` from the pair file; in C++ to be checked). The cheaper
`PartonLevel:MPI = off` loses the pairing (other Pythia events) and changes the fragments.

## 4. Alternatives (if steps 1–3 are not enough)

- **Smooth Cooper–Frye** on stored surfaces: no sampling noise at all; needs `--write-particlize`
  and an integrator with resonance decays (none in the repo). Only if step 2's noise still limits.
- **FastHydro's single-parton gun** (`contribs/FastHydro/example/make_hadron_wake_data.py`): one
  60 GeV parton from the centre at φ = 0, y = 0, so events stack without frame smearing; scans of
  `tau_delay`, `e_threshold`, η/s in minutes. Au+Au 200 GeV, FastHydro, ideal Cooper–Frye:
  shape sensitivities only, no CMS comparison.
- **Back to data**: rank measurable observables by their sensitivity per CMS-sized error with
  the truth maps, then pass the best through `mix.py`: Z+jet–hadron with Δη to the jet, a
  narrow Z-side Δφ window, even harmonics, missing p_T in ΔR rings.
