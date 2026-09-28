# analysis — jet energy balance and the LBT/liquefier double counting

This folder has the analysis notebooks for the jet productions, plus two scripts about one
question: **does the energy of the initial partons come back as surviving partons plus the
energy deposited in the medium?**

| script | what it is for | writes? |
|---|---|---|
| [`jet_edep_balance_check.py`](jet_edep_balance_check.py) | checks the balance and finds partons that are counted twice | no |
| [`fix_lbt_double_counting.py`](fix_lbt_double_counting.py) | repairs files made before the X-SCAPE fix, so they can be hadronized | yes, in place, reversibly |

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
