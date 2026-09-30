# analysis — jet energy balance, the LBT/liquefier double counting, wake observables

This folder has the analysis notebooks for the jet productions. §1–§5 are about one
question: **does the energy of the initial partons come back as surviving partons plus the
energy deposited in the medium?** §6 covers the parton- and hydro-level wake analysis, §7 the
same at hadron level, §8 the same distributions in ROOT.

| file | what it is for | writes? |
|---|---|---|
| [`jet_edep_balance_check.py`](jet_edep_balance_check.py) | checks the balance and finds partons that are counted twice | no |
| [`fix_lbt_double_counting.py`](fix_lbt_double_counting.py) | repairs files made before the X-SCAPE fix, so they can be hadronized | yes, in place, reversibly |
| [`wake_observables.py`](wake_observables.py) | one pass over a production → `wake_observables.h5` | a new file only |
| [`wake_observables.ipynb`](wake_observables.ipynb) | figures from `wake_observables.h5` only | no |
| [`wake_hadrons.py`](wake_hadrons.py) | one pass over a production's hadron files → `wake_hadrons.h5` | a new file only |
| [`wake_hadrons.ipynb`](wake_hadrons.ipynb) | hadron-level figures from `wake_hadrons.h5` + `wake_observables.h5` | no |
| [`hadron_distributions.C`](hadron_distributions.C) | ROOT macro: η, φ, pT of background, background + deposition, jet fragments and the wake, from the ROOT files of `run_h5toROOT.py` | a new `.root` + `.pdf`/`.png` |

Everything here except `hadron_distributions.C` runs without X-SCAPE, in the venv of
[`utils/analysis_env`](../../../../utils/analysis_env/README.md).

Both scripts read the pair files that `run_prod_jet.py` writes (`<stem>.h5`), and the fix
script also reads the `<stem>_particlize.h5` next to each one.

The balance was first checked in §11 of
[`../prod_AuAu_0_10_jet/jet_wake.ipynb`](../prod_AuAu_0_10_jet/jet_wake.ipynb); that is where
the bug was found. The notebook works one event at a time, and the scripts do the same checks
for a whole production.

> **Status (2026-09-28).** The fix is on X-SCAPE branch `fix_lbt_liquifer_double_counting`
> (commit `780f5036`), not yet merged into `contrib`. Every production made before it has the
> bug. `FNO_Hydro_Data/AuAu_0_10_pth10-40_eta06` has been repaired with
> `fix_lbt_double_counting.py` (766 partons in 266 of 300 events).

## 1. The balance

The liquefier moves energy from the shower into MUSIC_2. It should neither create any nor
lose any. Each final parton carries the status the liquefier gave it:

| pstat | parton | where its energy goes |
|---|---|---|
| 0, 1, 22 | shower parton, recoil, photon | survives and is hadronized |
| −1 | hole that was not absorbed (only heavy quarks and photons) | survives, hadronized with −E |
| −11 | fell below `e_threshold` (2 GeV in the fluid rest frame) | absorbed into a droplet |
| −13 | the four-momentum a vertex did not conserve | into a droplet |
| −17 | absorbed hole | into a droplet with −E |

A **hole** is the thermal parton a recoil was knocked out of: medium energy the recoil took
with it. LBT makes one in every scattering. MATTER makes them only with `<recoil_on> 1`.

At every vertex, `LiquefierBase::add_hydro_sources` deposits one droplet: (what came in) −
(what goes on). Summing over the shower graph (`shower/partons` in the pair file) gives
three identities:

- **(a) the graph closes:** $E_\text{ini} + E(-17) = E(0,1,22) - E(-1) + E(-11) + E(-13)$.
  Holes are inputs to the vertex they appear at, so they sit on the left.
- **(b) the droplet table is what the graph liquefied:**
  $\sum E_\text{droplet} = E(-11) + E(-13) - E(-17)$.
- **(c) energy is conserved:** $E_\text{ini} = E_\text{surviving} + \sum E_\text{droplet}$.

(c) follows from (a) and (b). Identity (a) holds to $10^{-13}$ GeV nearly by construction,
because the liquefier adds a −13 parton to every vertex that does not conserve. That makes
(a) a bookkeeping check. The −13 sum is what measures how far LBT is from conserving.

## 2. The bug

LBT free-streams a parton by returning it as the only outgoing parton (`pOutTemp.size() == 1`
in `JetEnergyLoss::DoExecTime`). The liquefier can drop that parton below `e_threshold` right
there, and it then deposits the parton's whole four-momentum as a droplet. But in that branch
`JetEnergyLoss` writes no new edge to the shower graph. So the parton's last edge kept
pstat 0 or 1, and the final-parton list still returned it as a survivor.

**The same energy was counted twice**: once in MUSIC_2 as a droplet, and again when
ColorlessHadronization fragmented the parton.

- **What was right:** the deposition. MUSIC_2 got the correct source, so the wake, Δe, the
  bulk hadrons from iSS and the freeze-out time are all fine.
- **What was wrong:** the surviving jet was too big. That affects jet fragments, jet pT and
  any energy-loss fraction taken from the final partons. Some events lose their whole jet to
  the medium; the buggy files still show surviving partons for them.
- **How big:** 3.0% of the initial energy for the seed-1 example (pTHat 50–70, one parton of
  3.77 GeV). In the pTHat 10–40 production, 766 partons in 266 of 300 events; there the
  surviving energy summed over events was 31% too high (42.0 instead of 32.0 GeV per event,
  against 41.6 GeV deposited).

In the identities of §1, the bug keeps (a) closed, because the parton is a pstat 1 leaf. But
(b) and (c) fail: the table holds a droplet that no vertex of the graph makes, and (c) counts
it twice.

## 3. The fix

On X-SCAPE branch `fix_lbt_liquifer_double_counting`, `JetEnergyLoss::MarkLiquefiedEdge` now
handles this case. When the liquefier absorbs a free-streaming parton, it sets −11 on that
parton's own graph edge, the edge ending at the parton's current vertex (holes are skipped).

Two runs were repeated with the same seeds: seed 1, and job 0001 of the pTHat 10–40
production (15 events). For both, the droplets, the initiators, both hydro legs and the
freeze-out times are bit-identical to the buggy runs. The only change is the pstat of the
double-counted partons, 0/1 → −11. So a run with the fixed build differs from an old run only
in those pstat values, which is why old files can be repaired.

## 4. The two scripts

### `jet_edep_balance_check.py` — the diagnostic

It checks (a), (b) and (c) for every event, and finds the partons counted twice. A
double-counted parton is a surviving final parton whose four-momentum equals a droplet of the
same event. Use it on any production to see whether it has the bug. It only reads.

```
python jet_edep_balance_check.py FILE.h5 [FILE.h5 ...] [--per-event]
```

A file made with the fixed build, or repaired, shows `surv + drop - ini` = 0 and
`double-counted partons: 0`. A buggy file shows a positive `surv + drop - ini` and names the
double-counted partons. In both cases `minus double-counted` must be ~1e-6 GeV, the float
precision of the droplet table. If it is not, something else is wrong.

### `fix_lbt_double_counting.py` — the repair

Rerunning a production with the fixed build would repeat all of its hydro. That is not
necessary: the hydro is already right, and only the pstat of the double-counted partons is
wrong. This script finds those partons as the diagnostic does, and sets them to −11 in two
places:

- `shower/partons` of the pair file (the shower graph);
- `partons/data` of the `_particlize.h5`, the table `hadronize.py` fragments.

The patched shower graph of job 0001 is bit-identical to the one the fixed build wrote for
the same seeds.

```
python fix_lbt_double_counting.py DIR_OR_PAIR_H5 [...]            # dry run: report only
python fix_lbt_double_counting.py DIR_OR_PAIR_H5 [...] --apply    # patch in place
python fix_lbt_double_counting.py DIR_OR_PAIR_H5 [...] --revert   # undo
```

Safeguards:

- nothing is written without `--apply`;
- the pair file and the particlize file must pick the same partons in every event, otherwise
  neither file is written;
- after patching, (c) must hold to $10^{-4}$ GeV in every event, otherwise the file is reported
  as failed;
- the changed rows and their old pstat are kept next to each table (`lbt_fix_rows`,
  `lbt_fix_pstat_before`), and the file gets the attribute `lbt_double_count_fix`. A file is
  therefore patched only once, and `--revert` restores it bit for bit.

The script rewrites only the parton tables (a few kB per file), so it takes about a second
for a 74 GB production.

## 5. Workflow for an old production

```
python jet_edep_balance_check.py DIR/*_[0-9][0-9][0-9][0-9].h5    # has it the bug?
python fix_lbt_double_counting.py DIR                              # what would change
python fix_lbt_double_counting.py DIR --apply
python jet_edep_balance_check.py DIR/*_[0-9][0-9][0-9][0-9].h5    # now 0 double-counted
cd ../prod_AuAu_0_10_jet && python hadronize.py DIR/<stem>_particlize.h5 ...
```

After a repair, only `jet_frag` changes. `bulk_jet` and `bulk_bg` come from the surfaces, so
existing bulk hadron files stay valid, and `hadronize.py --tags jet_frag` is enough.

**Comparing fragments with partons.** ColorlessHadronization closes a string that has no
partner quark with a fake beam-remnant quark. That quark carries $\sqrt{s_{NN}}/6$ = 33.3 GeV
along the beam (`eCMforHadronization` = 200 in `hadronize.xml`). So the energy of `jet_frag`
per sample is the surviving parton energy plus 33.3 GeV per remnant. Removing a parton can add
or remove a remnant. The remnant hadrons are very forward (pT ≈ 0.3 GeV), so a
midrapidity cut removes them; a sum over all η does not.

## 6. Wake observables: script, then notebook

The analysis is split so that a new figure never needs another pass over the data:

```
python wake_observables.py DIR -j 10 -o DIR/wake_observables.h5    # ~1 min for 300 events
jupyter lab wake_observables.ipynb                                 # WAKE_OBS=<file> to choose
```

[`wake_observables.py`](wake_observables.py) reads each pair file once and writes these
tables and arrays to one HDF5 file (a few MB):

- **`showers`:** per initial parton: kinematics; vertex; `cos_alpha` (heading in or out of the
  fireball); the angle to the background's ψ₂; the exact energy bookkeeping from the graph;
  deposition times. Also the medium along its straight path through the *background* leg: path
  length above T_c = 0.16 GeV, ∫T², ∫T³, ∫(τ−τ_in)T³ and their flow-weighted versions with
  γ(1 − v·n).
- **`droplets`:** position, four-momentum, the shower it came from (assigned exactly through the
  graph), background T and v at the deposit, and `kernel_flux` (see below).
- **`events`:** pTHat window and cross section, ψ₂ and ε₂, freeze-out times, energy totals.
- **`evolution`:** P^μ and S of both legs through every τ surface (ideal-fluid T^{μν} with the
  EoS table MUSIC used), and the cumulative droplet four-momentum.
- **`jetframe`:** the stacked wake about the leading-deposit leg's source, in (Δη, Δφ).

To add a figure, add a cell to the notebook. To add a quantity, add it to the script and rerun
it.

**Speed.** `-j` processes files in parallel; within a file, the time goes to reading the legs
(Blosc decompression of one chunk per τ frame) and to P^μ and S through every frame. The script
reads event k + 1 in a thread while it computes event k (h5py releases the GIL while it reads),
reads a reused background once, and finds the EoS interval without a search, since MUSIC's table
is uniform in e. On a gridnorm file (15 events, one background) that took one core from 17–19 s
to 10.5 s, with bit-identical output. The read-ahead holds one more event per job, ~0.5 GB with
its background: budget for it with a large `-j`.

**A second bug the analysis found: MUSIC_2 does not receive the droplets' energy.** The
CausalLiquefier kernel is point-sampled at MUSIC's cell centres in the one step that deposits a
droplet, and the sampled sum is not normalized to 1.

- The kernel is a ball of radius c_diff (t − t_d), seen at t − t_d ≈ cosh η_d τ_delay. In lab z
  one η cell is τ Δη cosh η long, so at large |η_s|, or late τ, the ball covers only a few cells.
- The script ports the kernel (`KernelFlux`) and computes each droplet's `kernel_flux`, the
  fraction MUSIC_2 actually receives. It ranges from ~0 to ~6.
- In `AuAu_0_10_pth10-40_eta06` the injected energy is off by more than 20% in 40% of events.
- The wake's ΔP^μ follows the kernel-weighted deposit (to ~2% until τ ≈ 8 fm), not the droplet
  energy.

This is independent of the LBT double counting. A second, smaller loss has the same effect:
droplets at the hard vertex (τ_d = 0) were stored with η = 0/0, and the C++ kernel is then NaN
everywhere, so MUSIC never received them (13 droplets, 56 GeV, in the pTHat 10–40 production).

**The fix** (X-SCAPE branch `liquifier_kernel_normalization`, `<Liquefier>
<normalize_on_hydro_grid>`, on by default):

- `MusicWrapper` passes MUSIC's grid to the liquefier.
- Before a droplet's deposit step, `LiquefierBase` computes its sampled kernel sum on that grid
  and scales the droplet by its inverse. A droplet whose kernel misses every cell centre goes
  whole into its nearest cell.
- Vertex droplets get η = 0.

Rerunning job 0003 of the pTHat 10–40 production with the same seeds:

- the droplets, showers and background leg are bit-identical;
- the wake follows the droplets themselves: ΔP⁰/droplets at τ = 7.5 fm has median 1.02 and
  16–84% 0.95–1.06, against 0.70–1.11 before.

Pair files made with the fix have the root attribute `liquefier_normalize_on_hydro_grid = 1`
and each droplet's C++ kernel sum in `source/flux`. `wake_observables.py` then sets
`inj_factor` = 1, i.e. MUSIC received each droplet exactly. For older files it uses the Python
port (0 for vertex droplets), and the notebook's hydro-level results are normalized to the
kernel-weighted deposit `E_inj_hydro`.

Productions made before the fix cannot be repaired afterwards: their jet leg (`arr`) was
evolved with the wrong source. Rerun them; with the same seeds, only `arr` changes.

### What the fix changes, and where it shows up

The fix only changes what MUSIC_2 receives. For the same seeds, the droplets, the showers
and the background leg are bit-identical.

**Unchanged:** every parton-level quantity. That includes the surviving jet energy, ΔE vs pT,
geometry, path length, A_J, deposition times and temperatures, the background hadrons
(`bulk_bg`) and the jet fragments (`jet_frag`). Two productions made with different seeds
differ here only statistically.

**Changed:** the jet leg `arr`, i.e. the wake Δe, and everything built from it: the freeze-out
surface of the jet leg, the `bulk_jet` hadrons, the freeze-out delay and FNO training pairs.
The errors sit in three places.

1. **Event by event.** Before the fix, 40% of events had their injected energy off by more
   than 20%, with single droplets from 0.004× to 6×. The correlation of the wake's ΔP⁰ with
   its own droplets at τ = 7.5 fm was 0.55; with the fix it is 0.97. Anything that relates a
   wake to its own event was smeared: wake size against the jet's energy loss, the
   freeze-out delay against the deposit (its spread falls from 0.88 to 0.66 fm between the
   two productions), and the bulk energy recovered at hadron level per event.
2. **At large |η_s|.** Most of the error is there: droplets at |η_s| > 1 carry 38% of the
   deposited energy and had fluxes from 0 to 6. Job 0003, same seeds, 15 events, wake ΔP⁰ at
   τ = 7.5 fm by |η_s| band [GeV] (the wake spreads in η after the deposit, so its bands are
   not the droplets' bands):

   | \|η_s\| | droplets deposited | old wake | fixed wake | Σ \|fixed − old\| per cell |
   |---|---|---|---|---|
   | 0 – 0.5 | 342 | 285 | 291 | 36 |
   | 0.5 – 1 | 105 | 172 | 180 | 60 |
   | 1 – 2 | 120 | 130 | 122 | 152 |
   | 2 – 5 | 86 | 53 | 76 | 217 |

   Near midrapidity the wake changes by about 2%. Forward, the energy is in the wrong
   places: the cell-by-cell difference is larger than the whole wake there. This affects the
   longitudinal shape of the wake, wide-Δη correlations, and hadrons from the forward part
   of the surface.
3. **Late deposits.** Droplets deposited after τ ≈ 7 fm had fluxes far below 1 (down to
   0.013), so the late wake, and the reheating that extends freeze-out when a deposit comes
   near the end, were underestimated.

**For FNO training** this matters most. The model learns the map from droplets to Δe. Before
the fix, Δe did not correspond to the droplets the model is given, especially at forward
rapidity and late τ. That is noise it cannot learn away, and it is biased forward. The fixed
productions give consistent pairs.

**Nearly unchanged on average:** ensemble averages at midrapidity (the old energy-weighted
mean flux was 0.97). Across the two 300-event productions:

- the fraction of the stacked wake at |Δη| > 1 is 0.29 (old) against 0.28 (fixed);
- the entropy per wake energy is 3.26 against 3.20 GeV⁻¹.

So the midrapidity jet-frame maps and Δφ profiles of the old production still hold
qualitatively.

**The check across productions.** Wake ΔP⁰ against the droplets deposited by τ = 7.5 fm:

| | median | 16–84% | correlation |
|---|---|---|---|
| `AuAu_0_10_pth10-40_eta06` (old) | 0.93 | 0.65 – 1.10 | 0.55 |
| `AuAu_0_10_pth10-40_eta06_gridnorm` (fixed), 285 events | 0.99 | 0.91 – 1.06 | 0.97 |

**Remaining limit:** normalization fixes the integral, not the shape. A hole concentrated on
a few cells at large |η_s| can push a cell below zero energy. MUSIC then resets it
(`reconst.cpp`), which adds energy, and that is where the remaining tails come from
(2.5–97.5%: 0.74–1.21).

### Legs that never froze out

In `AuAu_0_10_pth10-40_eta06_gridnorm`, job 0002's background (MUSIC_1, no jet source)
stops updating part of its grid at τ ≈ 4.6 fm. Its maximum e stays at 2.8 GeV/fm³, so it
never freezes out: MUSIC runs to its maximum time and logs "Maximum allowed time reached".
The leg's ΔP is then meaningless: about −6×10⁴ GeV at the "last live frame". This is a
MUSIC4GPU problem with that initial condition. It is reproducible, and identical with X-SCAPE
`contrib`, so it has nothing to do with the liquefier.

`wake_observables.py` records each leg's hottest cell in its last frame and flags
`jet_no_freezeout` / `bg_no_freezeout` when it is above 1.5 e_fo. The notebook leaves those
events out of the hydro-level figures and names the files. To find such jobs:
`grep -l "Maximum allowed time reached" DIR/*.log`.

## 7. The wake at hadron level: script, then notebook

The same split as §6, for the hadron files `hadronize.py` writes (`bulk_jet`, `bulk_bg`,
`jet_frag`):

```
python wake_hadrons.py DIR -j 10 -o DIR/wake_hadrons.h5    # ~3.5 min for 300 events, ~30 MB (common seeds: ~190 MB / 1000 ev)
jupyter lab wake_hadrons.ipynb                             # WAKE_HAD=<file>, WAKE_OBS=<file>
```

[`wake_hadrons.py`](wake_hadrons.py) reads each production file's three hadron files once.
For every event it stores per-event sample means, each with the variance of the mean, taken
from the spread of the per-sample sums:

- the jet leg;
- that event's background, binned in the event's own frame;
- the fragments.

The frame is the leading initiator parton. There are four histograms:

- `jetframe`: (Δη, Δφ, pT) of charged hadrons;
- `dR`: ΔR profiles;
- `spectra`: identified spectra at |y| < 1, near and away side;
- `totals`: N, E, pT, pT cos Δφ, pz, p·n per |η| band and pT.

The wake is jet leg − background, event by event. The notebook joins the file with
`wake_observables.h5` for the deposit, the showers and the freeze-out times.

**Common seeds.** With `hadronize.py --common-seeds` or `--correlated`, the jet legs of the
events sharing a background draw its random numbers sample by sample. With `--correlated`
they also share most of their hadrons with it. So neither the jet leg nor the wake is
independent from event to event. For such files the script also stores the jet leg and the
wake (jet − bg) as `--batches` (default 10) means over aligned blocks of samples, in
`hist/<name>/{jet,wake}/batch`. The notebook sums them over the events of each background and
takes the error of the bkg + wake and wake curves from their spread. That spread holds all
these correlations, across bins too. The bkg and fragment errors are computed as before.

The batch means are rounded to `--batch-bits` (default 10) mantissa bits, a relative error
below 5e-4 per value. The difference jet − bg is taken before rounding. Rounding the two legs
separately would not work: where they share most hadrons, the wake is much smaller than
either leg, and the legs' rounding errors would swamp its noise (up to 7× in single
`jetframe` bins).

Errors for `AuAu_0_10_pth10-40_eta06_c1` (1005 events, 15 per background, 100 samples per
leg, `--correlated`), per event:

| | before (legs added) | batch means |
|---|---|---|
| wake E | ±1.21 GeV | ±0.16 GeV |
| wake N_ch at \|y\| < 1 | ±0.51 | ±0.12 |
| wake N_ch, near side (\|Δφ\| < π/3, \|Δη\| < 1) | ±0.27 | ±0.04 |
| wake behind the leading jet, 0–1 GeV (leading leg < 30%) | ±0.32 | ±0.06 |
| bkg + wake N_ch at \|η\| < 1 | ±0.12 (too small) | ±0.48 |

The means don't change. The wake errors drop 4–8× and now sit below the file-bootstrap
errors. The bkg + wake error rises about √15, since the 15 jet legs of a background are nearly
fully correlated. Against `HadronFileReader`'s exact per-sample errors, the batch means agree
within 2% (wake ±0.111 vs ±0.109, bkg + wake ±0.4806 vs ±0.4807, N_ch at \|η\| < 1). Rounding
to 10 bits changes the errors by less than 6e-4 in every bin, and no number the notebook
prints.

File size for the same campaign:

| `wake_hadrons.h5` | size |
|---|---|
| without batches (version 1, or independent files) | 59 MB |
| with batches, lossless (`--batch-bits 23`) | 254 MB |
| with batches, 10 bits (default) | 189 MB |
| with `--batches 5` and 10 bits (estimate) | ~125 MB |

`jetframe` (3840 bins per event) is two thirds of the batches. Independently sampled files get
no batches: their files and results are unchanged.

**Stored hadrons.** `hadronize.py --eta-max` and `--charged` store only part of the hadrons,
and every histogram then holds only that part. The script records the cut per event
(`events/eta_max`, `events/charged_only`) and warns, and so does the notebook. The c1 campaign
keeps |η| < 2. Its "all η" rows are therefore |η| < 2, and the energy balance misses the
forward hadrons and ColorlessHadronization's beam remnants. The jet frame is also cut at
|Δη| ≈ 2 − |η_jet|.

**Bad runs.** Before loading any hadrons, the script runs these checks on every event:

- the freeze-out test of `wake_observables.py`, from the pair file (the hottest cell of a
  leg's last frame is above 1.5 e_fo);
- a hadron-level test: the background's multiplicity per sample and its freeze-out cells must
  be within 20% of the jet leg's (the wake changes them by < 1%);
- the log's "Maximum allowed time reached" (reported only);
- a cross-check against `wake_observables.h5`.

Flagged events are not histogrammed (`usable` = False), unless `--keep-flagged` is given. In
`AuAu_0_10_pth10-40_eta06_gridnorm` exactly job 0002 fails, on every check. Its background has
11× the jet leg's hadrons and 3.2× its cells. Good jobs are within 0.7% and 1.8%.

**Beam remnants.** ColorlessHadronization closes colour-unpaired strings with a remnant of
√s/6 = 33.3 GeV (§5). The fragments' energy is E_surv + n_rem √s/6 with an integer n_rem per
event (0–2 here), the same in every fragmentation. The notebook derives n_rem from that and
takes the remnants out of the energy balance. Their strings put fragments at all Δη, so no η
cut removes them.

**Results for `AuAu_0_10_pth10-40_eta06_gridnorm`** (285 events; details in §9 of the notebook):

- the wake's hadrons carry 1.01 ± 0.13 ± 0.18 of the energy MUSIC_2 received;
- fragments − remnants + wake = 1.006 E_ini;
- the wake has thermal chemistry and a harder spectrum than the background (⟨pT⟩ 0.85 against
  0.54 GeV);
- it peaks along both jets, with no significant depletion behind them;
- it brings back 20% of the leading leg's deposit inside R = 0.4 and 70% inside 1.5;
- it balances the fragments' pT with soft hadrons at 0.5–2 GeV.


## 8. Hadron distributions in ROOT: `hadron_distributions.C`

A ROOT macro for the ROOT files of
[`../prod_AuAu_0_10_jet/run_h5toROOT.py`](../prod_AuAu_0_10_jet/run_h5toROOT.py) (layout in
[`../prod_AuAu_0_10_jet/root_export/README.md`](../prod_AuAu_0_10_jet/root_export/README.md)).
It histograms η, φ and pT for each source of a jet event:

| name | ROOT ntuple | what |
|---|---|---|
| `bkg` | `bulk_bg` | iSS on the background's surface (MUSIC_1) |
| `bkgdep` | `bulk_jet` | iSS on the jet leg's surface (MUSIC_2): background + deposition |
| `frag` | `jet_frag` | ColorlessHadronization of the surviving partons |
| `wake` | | `bkgdep` − `bkg` |
| `full` | | `bkgdep` + `frag`, the whole jet event |

```bash
cd example/analysis
root -l -b -q 'hadron_distributions.C+("DIR")'                 # per event, all windows
root -l -b -q 'hadron_distributions.C+("DIR", 2)'              # pT-hat window 2 only
root -l -b -q 'hadron_distributions.C+("DIR", -1, true)'       # cross-section weighted [mb]
root -l -b -q 'hadron_distributions.C+("DIR", -1, false, false, 1.0, "out.root", 8)'
         # all hadrons (not only charged), |eta| < 1 for pT and phi, output file, 8 threads
```

Arguments: `dir, window = -1, xsec = false, charged = true, eta_cut = 1.0, out =
"hadron_distributions.root", threads = 1, keep_flagged = false`. η is filled for all pT;
pT and φ for \|η\| < `eta_cut`. The output holds `h_{eta,phi,pt,ptlog}_{bkg,bkgdep,frag,wake,full}`,
divided by the bin width, plus one page of plots (`.pdf`, `.png`). The rows of that page are
background vs background + deposition, fragments, and wake.

**Normalization is `HadronFileReader`'s.**
- **Per event.** Each event is the mean over its oversamples, so a hadron of event e
  weighs w_e / n_samples. A background shared by several events counts once for each of
  them.
- **Per event** (default): w_e = 1/N, so the histograms are dN/dX per event.
- **`xsec`**: w_e = σ_k / N_k, the window's cross section from the campaign file over its
  selected events, so the sum over windows is dσ/dX in mb. N_k counts only the events used,
  so dropping flagged events does not change the cross section.
- **Errors** are the compound-Poisson errors of independent sampling. For `--correlated`
  hadron files the wake's paired error is much smaller: use
  `HadronFileReader.jet_minus_background` for it.

**Bad runs** are dropped as in `wake_hadrons.py`: an event whose background has more than
20% more or fewer freeze-out cells than its jet leg (`events.n_cells_bg / n_cells_jet`) is
flagged and left out, unless `keep_flagged`. In `AuAu_0_10_pth10-40_eta06_gridnorm` those
are the 15 events of job 0002.

**Checked** on `AuAu_0_10_pth10-40_eta06_gridnorm`, 285 events, charged hadrons at
\|η\| < 1: every pT bin of `bkg`, `bkgdep`, `frag` and `wake`, and its error, equals
`HadronFileReader.hist` / `jet_minus_background` on the HDF5 files to 10⁻¹². Per event:
1308 (bkg), 1317 (bkg + deposition), 7.0 (fragments), 9.0 (wake) charged hadrons. The
campaign takes 21 s with 8 threads, against ~6 min for the same histograms with
`HadronFileReader` on the HDF5 files.
