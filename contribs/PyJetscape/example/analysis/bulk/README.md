# bulk — minimal flow and bulk observables

Per event charged multiplicity, mean transverse momentum and two-particle flow with an η gap,
for the whole event, its parts and the paired difference to the jet-free background; class
averages with jackknife errors. This is the minimal set of
[`PLAN_multi_hard.md`](../../../../../docs/Plans/PLAN_multi_hard.md) (D15, §3.10), enough for
its check V5: what the extra hard scatterings (and the trigger's wake) do to Nch, [p_T] and
v_n{2}, in Pb+Pb, Au+Au, O+O, Ne+Ne and minimum-bias files. How to produce those files and
what the extras are: [`docs/README_extra_hard.md`](../../../../../docs/README_extra_hard.md).

| file | |
|---|---|
| [`flow.py`](flow.py) | the library: `Cuts`, `event_table(reader, cuts)`, `select(...)`, `class_averages(table, mask)`; the definitions are in its docstring |
| [`bulk_obs.py`](bulk_obs.py) | the command line: a table per N_coll class, `--out` (.json / .npz), `--table` (per-event .npz) |
| [`../../../tests/test_bulk_flow.py`](../../../tests/test_bulk_flow.py) | synthetic events with an imposed v₂, v₃ and [p_T] spread |

```sh
python bulk_obs.py out_had/                                   # every production file in it
python bulk_obs.py "OUTBASE/gpu*/*_hadrons_bulk_jet.h5" --ncoll-classes 0 10 30 50 100 \
       --out bulk.json --table bulk_events.npz
python bulk_obs.py out_had/ --eta-max 0.5 --flow-eta-max 2.4 --eta-gap 2 --harmonics 2 3 4
```

```python
from flow import Cuts, event_table, select, class_averages
with HadronFileReader("out_had") as r:
    t = event_table(r, Cuts())
res = class_averages(t, select(t, n_coll_percent=(0, 10)))
res["full/v2"], res["delta_full/dnch"]           # (value, jackknife error)
```

## What is computed

Charged hadrons (|pid| in `hadrons_h5.CHARGED`, which includes e and μ: a Z's leptons are
above the default p_T range), 0.2 < p_T < 3 GeV, by default:

| per event | definition |
|---|---|
| `nch` | hadrons with \|η\| < 0.8, the mean over the oversamples |
| `pt` | [p_T] = Σp_T / N in that window, per oversample, then the mean over the oversamples |
| `pt_var` | the variance of [p_T] over the oversamples (the sampling noise of one sample) |
| `c`, `w` | the two-subevent pair sum Re(Q_n^A Q_n^B*) and its pair count M_A M_B, means over the oversamples; A: −η_max^flow < η < −Δη/2, B: Δη/2 < η < η_max^flow (default 0.5 < \|η\| < 0.8, the standard \|Δη\| > 1) |

for four **compositions**: `full` = bulk_jet + jet_frag + extra_frag (the event, D16),
`no_extras` = bulk_jet + jet_frag, `bulk_jet` (bulk and every wake), `bulk_bg` (the jet-free
twin). Minimum-bias files have no jet_frag, files without extras no extra_frag: those parts
are then empty (full = no_extras without extras). The fragmentation used with oversample k is
k mod n_frag, as `HadronFileReader.jet_event` takes it. Each event also records N_coll, the
number of extras and their summed E and p_T (from extra_frag's `scatterings/`).

| per class | definition |
|---|---|
| `nch`, `dnch_deta` | ⟨nch⟩ and ⟨nch⟩ / (2 η_max) |
| `pt` | ⟨[p_T]⟩ over the events |
| `sigma_pt` | the spread of the oversample-averaged [p_T] over the events; still contains each average's noise |
| `sigma_pt_dyn` | √(sigma_pt² − ⟨pt_var / K⟩): the event-by-event fluctuation of the fluid's [p_T], the sampling noise removed with the oversamples (NaN with one oversample) |
| `sigma_pt_one` | √(sigma_pt_dyn² + ⟨pt_var⟩): the spread with one sample per event, as an experiment sees it |
| `c2`, `c3` | ⟨⟨2⟩⟩_n = Σ_events c / Σ_events w (multiplicity-weighted) |
| `v2`, `v3` | v_n{2, \|Δη\| > 1} = √⟨⟨2⟩⟩_n (NaN if ⟨⟨2⟩⟩_n ≤ 0) |

and the **paired difference** to the background, `delta_full`, `delta_no_extras`,
`delta_bulk_jet` (composition − bulk_bg, over the events that have both):

| | definition |
|---|---|
| `dnch`, `rdnch` | ⟨nch − nch_bg⟩ and that over ⟨nch_bg⟩ |
| `dpt` | ⟨[p_T] − [p_T]_bg⟩ |
| `dc2`, `dc3` | ⟨⟨2⟩⟩_n − ⟨⟨2⟩⟩_n,bg: the change of the correlator (additive, defined when v_n{2} is not) |
| `dv2`, `dv3` | v_n{2} − v_n{2}_bg |

and **the flow vector's kick** per paired difference: with q_n = Q_n / M the per-hadron flow
vector of a subevent and Δq = q(composition) − q(bulk_bg), sample k against sample k,

| | definition |
|---|---|
| `kick2{n}`, `kick{n}` | ⟨Re(Δq_n^A Δq_n^B*)⟩ = ⟨\|ΔV_n\|²⟩ and its root: how far the event's flow vector moves (size and direction); unbiased by the sampling noise, which is independent in A and B |
| `rkick{n}` | kick / v_n{2} of bulk_bg |
| `align{n}` | ⟨Re(q_bg^A Δq^B* + Δq^A q_bg^B*)⟩ = 2⟨V·ΔV⟩: the change along the existing flow (dilution, push); a random kick adds nothing here |
| `r{n}` | ⟨Re(q^A q_bg^B*)⟩ / √(⟨Re(q^A q^B*)⟩ ⟨Re(q_bg^A q_bg^B*)⟩): the correlation of the flow vectors with and without (1: untouched) |

The per-hadron ⟨⟨2⟩⟩ changes by exactly align + kick2, event by event. These weigh every
event the same (per-hadron vectors). kick is NaN where kick2 ≤ 0 (consistent with no kick).

With D13's minimum-bias mode bulk_jet − bulk_bg is the extras' effect (their wakes; `full`
adds their fragments); in a triggered file it is the trigger's and the extras' together.

## Conventions

* **Oversamples** are independent Cooper-Frye samplings of the same fluid, not independent
  collisions. Every oversample is treated as one physical event (pairs and [p_T] within one
  sample, decays' non-flow included as in data), the per-event values are means over the
  oversamples, and every event weighs the same whatever its number of oversamples. The legs
  may have different numbers of oversamples: only per-event means are differenced.
* **Paired difference.** Both legs are averaged over the same events and their difference
  is taken per jackknife replica, so the errors keep the legs' correlation. With
  `hadronize.py --correlated` sample k of bulk_jet and of its bulk_bg share their random
  numbers and Δ is low-noise; without it (e.g. `--hadrons-only` jobs so far) the legs are
  independent samplings and the error is the larger one of two independent samples.
* **Errors**: delete-one-block jackknife. The unit is the hydro background, not the event:
  events sharing a background (`--reuse`) share their bulk and are left out together (with
  `--reuse 1` the unit is the event); more than `--max-blocks` (100) units are grouped into
  contiguous blocks. A reused background counts once per event, as the readers count it.
  `n_events` entries have no error.
* **Classes**: a list of events (`--events`), an N_coll range (`--ncoll-min/max`), or N_coll
  percentiles (`--ncoll-classes`, the events ordered by N_coll, largest first) — a
  centrality-like proxy from the Glauber collision, not multiplicity classes.
* Hadron files stored with `--eta-max` smaller than the cuts' reach are refused.
* Every event weighs the same: right for minimum-bias files and single-window triggered
  files; `--pthat-bins` campaigns would need the window weights (not done).

## Limits (a later project, D15)

v_n{4} and higher cumulants, ρ(v₂², [p_T]), symmetry planes and event-plane correlations,
multiplicity-based centrality, p_T-differential v_n(p_T), the experiments' two-particle
[p_T] correlator ⟨Δp_T Δp_T⟩ (here the noise is removed with the oversamples instead),
cross-oversample pairs (A from one oversample, B from another: the fluid's v_n without the
decays' non-flow), event weights, and a C++ twin for the ROOT path.
