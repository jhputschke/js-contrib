# Bulk and flow

Work in progress

The analysis scripts and notebooks on this page are work in progress and need further vetting; they are meant mainly as examples of how to use the simulation outputs. See the [note on the analysis section](https://jhputschke.github.io/js-contrib/analysis/index.md).

The bulk tools in [`example/analysis/bulk/`](https://github.com/jhputschke/js-contrib/tree/main/contribs/PyJetscape/example/analysis/bulk) compute, per event, the charged multiplicity Nch, the mean transverse momentum [pT] and the two-particle flow vₙ{2} with an η gap. They do it for the full event, for its parts, and as the paired difference to the jet-free background, with class averages and jackknife errors. They work on every production with hadron files: Au+Au, Pb+Pb, O+O, Ne+Ne, with or without a trigger, with or without [extra hard scatterings](https://jhputschke.github.io/js-contrib/run/extra-hard/index.md). Their main use is the question the minimum-bias mode was made for: what do the extra hard scatterings (and a trigger's wake) do to Nch, [pT] and vₙ?

Like the other readers they need only the [analysis environment](https://jhputschke.github.io/js-contrib/install/analysis-env/index.md).

## 1. What is computed

Charged hadrons (`hadrons_h5.CHARGED`, which includes e and μ), 0.2 < pT < 3 GeV by default. Each event is evaluated for four **compositions**:

| composition | hadrons                                                 |
| ----------- | ------------------------------------------------------- |
| `full`      | `bulk_jet` + `jet_frag` + `extra_frag`: the whole event |
| `no_extras` | `bulk_jet` + `jet_frag`                                 |
| `bulk_jet`  | the bulk with every wake                                |
| `bulk_bg`   | the jet-free twin                                       |

A missing part is empty: without extras `full` = `no_extras`; minimum-bias files have no `jet_frag`.

| per event | definition                                                                                                                                                  |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nch`     | hadrons with (                                                                                                                                              |
| `pt`      | [pT] = ΣpT / N in that window, per oversample, then the mean over the oversamples                                                                           |
| `pt_var`  | the variance of [pT] over the oversamples (one sample's noise)                                                                                              |
| `c`, `w`  | the two-subevent pair sum Re(Qₙᴬ Qₙᴮ\*) and its pair count Mᴬ Mᴮ, means over the oversamples; A: −η_flow < η < −Δη/2, B: Δη/2 < η < η_flow (default 0.5 < ( |

| per class          | definition                                                                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `nch`, `dnch_deta` | ⟨nch⟩ and ⟨nch⟩ / (2 η_max)                                                                                                                      |
| `pt`               | ⟨[pT]⟩                                                                                                                                           |
| `sigma_pt`         | the spread of the events' [pT] (still contains each average's sampling noise)                                                                    |
| `sigma_pt_dyn`     | √(sigma_pt² − ⟨pt_var / K⟩): the fluid's event-by-event [pT] fluctuation, with the sampling noise removed using the K oversamples (NaN with one) |
| `sigma_pt_one`     | √(sigma_pt_dyn² + ⟨pt_var⟩): the spread with one sample per event, as an experiment sees it                                                      |
| `c2`, `c3`         | ⟨⟨2⟩⟩ₙ = Σ c / Σ w (multiplicity-weighted)                                                                                                       |
| `v2`, `v3`         | vₙ{2, (                                                                                                                                          |

And the **paired differences** to the background, `delta_full`, `delta_no_extras` and `delta_bulk_jet` (composition − `bulk_bg`, over the events that have both):

|                 | definition                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------ |
| `dnch`, `rdnch` | ⟨nch − nch_bg⟩, and that over ⟨nch_bg⟩                                                           |
| `dpt`           | ⟨[pT] − [pT]\_bg⟩                                                                                |
| `dc2`, `dc3`    | ⟨⟨2⟩⟩ₙ − ⟨⟨2⟩⟩ₙ,bg: the change of the correlator; it adds up, and it exists where vₙ{2} does not |
| `dv2`, `dv3`    | vₙ{2} − vₙ{2}\_bg                                                                                |

**The flow vector's kick.** The extras also move each event's flow vector in a random direction. Per paired difference the tools give `kick{n}` = √⟨|ΔVₙ|²⟩, measured as ⟨Re(ΔqₙᴬΔqₙᴮ\*)⟩ across the η gap so that the sampling noise drops out; `rkick{n}` (relative to vₙ{2}); `align{n}` = 2⟨V·ΔV⟩, the change along the existing flow (dilution or push); and `r{n}`, the correlation of the flow vectors with and without the extras. The per-hadron ⟨⟨2⟩⟩ changes by exactly align + kick², so the random and the systematic part come apart. On the 60-event Pb+Pb 0–10 % pilot (p₀ = 15 GeV): kick₂² = (3.1 ± 3.0)·10⁻⁶ (|ΔV₂| ≈ 0.002, ~4 % of v₂), kick₃² consistent with 0, r₂ = 0.9991 ± 0.0009.

**What a difference means** depends on the production:

- **Minimum bias** (`--hard none --extra-hard`): `delta_bulk_jet` is the extras' wakes, `delta_full` their wakes and fragments.
- **Triggered with extras:** the trigger's and the extras' together. Compare with the same seeds run without `--extra-hard` (the trigger and the background are bit-identical) to separate them.
- **Without extras:** the trigger's wake (`delta_bulk_jet`) and fragments (`delta_full`).

Each event also records N_coll, the number of extras and their summed E and pT (from `extra_frag`'s `scatterings/`).

## 2. Statistics

- **Oversamples** are independent Cooper–Frye samplings of the same fluid, not independent collisions ([What an event is](https://jhputschke.github.io/js-contrib/analysis/files/#what-an-event-is)). Each is treated as one physical event: pairs and [pT] are formed within one sample, so decay non-flow is included, as in data. The per-event values are means over the oversamples, and every event weighs the same whatever its number of oversamples. The legs may have different numbers of oversamples, since only per-event means are differenced.
- **Paired difference:** both legs are averaged over the same events and differenced in every jackknife replica, so the errors keep the legs' correlation. With correlated sampling (`--correlated`) sample k of `bulk_jet` and of its `bulk_bg` share their random numbers and the difference is low-noise. Without it the legs are independent samplings, and a difference of a few per mille needs many samples.
- **Errors:** delete-one-block jackknife over **backgrounds**, not events. Events sharing a background (`--reuse`) are left out together; with `--reuse 1` the unit is the event. More than `--max-blocks` (100) units are grouped into contiguous blocks.
- **Classes:** a list of events, an N_coll range, or N_coll percentiles (largest first). This is a centrality proxy from the Glauber collision, not multiplicity classes, which matter in small systems.
- **Weights:** every event weighs the same. That is right for minimum-bias and single-window files; `--pthat-bins` campaigns would need the window weights, which are not applied.
- Hadron files written with an `--eta-max` smaller than the cuts reach are refused.

## 3. The command line

```bash
cd contribs/PyJetscape/example/analysis/bulk
python bulk_obs.py out_had/                                  # every production file in it
python bulk_obs.py "OUTBASE/gpu*/*_hadrons_bulk_jet.h5" --ncoll-classes 0 10 30 50 100 \
       --out bulk.json --table bulk_events.npz
python bulk_obs.py out_had/ --eta-max 0.5 --flow-eta-max 2.4 --eta-gap 2 --harmonics 2 3 4
python bulk_obs.py out_had/ --workers 8                     # 8 processes, the same results
```

| option                       | default     |                                                                              |
| ---------------------------- | ----------- | ---------------------------------------------------------------------------- |
| `source ...`                 |             | directories, quoted globs or files (several allowed)                         |
| `--eta-max`                  | 0.8         | (                                                                            |
| `--pt-min`, `--pt-max`       | 0.2, 3.0    | GeV                                                                          |
| `--eta-gap`                  | 1.0         | Δη between the two subevents                                                 |
| `--flow-eta-max`             | `--eta-max` | outer edge of the subevents                                                  |
| `--harmonics`                | 2 3         | n of vₙ{2}                                                                   |
| `--events`                   | all         | global events, `0:100` or `3,5,9`                                            |
| `--ncoll-min`, `--ncoll-max` |             | an N_coll range                                                              |
| `--ncoll-classes`            |             | N_coll percentile edges, e.g. `0 10 30 50 100`                               |
| `--max-blocks`               | 100         | jackknife blocks (at most)                                                   |
| `--out`                      |             | the class results, `.json` or `.npz`                                         |
| `--table`                    |             | the per-event table, `.npz`                                                  |
| `--quiet`                    |             | no table on the screen                                                       |
| `--workers`                  | 1           | processes for the per-event pass (0: all CPUs); the same results as with one |

**In parallel.** The per-event pass is the slow part; `--workers N` (in Python `event_table(..., workers=N)`) reads the events in N processes, in contiguous chunks cut where the hydro background changes, so a shared background is still read once. The table, and so every result, is the same as with one process. On the GB10, 200 Pb+Pb 0–10 % events with 40 oversamples took 87 s with one process and 14 s with `--workers 8`. Scripts that call `event_table` with workers need the `if __name__ == "__main__":` guard (the processes are spawned).

The output on a 3-event minimum-bias O+O test job (one oversample, uncorrelated, so σ_dyn is NaN and the vₙ are noise; it shows the layout, not physics):

```text
1 production file(s), 3 events, tags bulk_jet, bulk_bg, extra_frag
  no jet_frag in (some) files (minimum bias): full = bulk_jet + extra_frag
  bulk_jet / bulk_bg sampled correlated: False
  oversamples per event: bulk_jet [1], bulk_bg [1];  N_coll 36-43;  extras per event 2.33, extra_E 927 GeV

== all: 3 events, 3 jackknife blocks
                       <Nch>            dNch/deta          <[pT]> GeV        sigma([pT])   ...   v2{2}
full                201.7 +- 26          126 +- 16       0.6656 +- 0.019   0.03209 +- 0.017 ...  0.08748 +- 0.029
bulk_bg             197.7 +- 19        123.5 +- 12       0.6524 +- 0.019   0.03325 +- 0.024 ...  nan
                        dNch           dNch/Nch_bg        d<[pT]> GeV        dc2 = d<<2>>  ...
delta_full              4 +- 15      0.02024 +- 0.078   0.01324 +- 0.0087  0.01226 +- 0.0038 ...
```

`--out bulk.json` holds `cuts`, `sources`, `tags`, `extras`, `correlated` and `classes`: per class, per composition and per difference, every quantity as (value, jackknife error).

## 4. In Python

```python
import sys
sys.path.insert(0, "contribs/PyJetscape/example/analysis/bulk")
from flow import Cuts, event_table, select, class_averages
from jetscape.hadrons_h5 import HadronFileReader

with HadronFileReader("out_had") as r:
    t = event_table(r, Cuts(eta_max=0.8, pt_min=0.2, pt_max=3.0, eta_gap=1.0))
    # t = event_table(r, Cuts(), workers=8)    # the same table from 8 processes
res = class_averages(t, select(t, n_coll_percent=(0, 10)))
res["full/v2"], res["bulk_bg/v2"]               # (value, jackknife error)
res["delta_full/dnch"], res["delta_bulk_jet/dc2"]
```

`flow.py` is the library (`Cuts`, `event_table`, `select`, `class_averages`; the definitions are in its docstring), `bulk_obs.py` its command line. The tests on synthetic events recover an imposed v₂ = 0.08 (0.0802 ± 0.0011), v₃ = 0.04, [pT] and its dynamical spread.

## 5. A recipe: what the extras do to the bulk

Minimum bias, correlated legs, a few oversamples, then the paired differences per N_coll class:

```bash
cd contribs/PyJetscape/example/prod_AuAu_0_10_jet
TAG_PREFIX=PbPb_0_10_seed ./run_jobs.sh -j 4 --campaign mb_x 20 30 0 out_mb_x \
    --user-xml ../config/PbPb_0_10_jet.xml --grid ../config/grid_PbPb.yaml \
    --hard none --extra-hard --hadrons-only --oversample 10 --correlated
cd ../analysis/bulk
python bulk_obs.py ../../prod_AuAu_0_10_jet/out_mb_x --ncoll-classes 0 50 100 --out mb_x.json
# delta_bulk_jet/dc2, dpt, rdnch: the extras' wakes;  delta_full: + their fragments
```

Scale the threshold to see the dependence on p₀ (`--extra-pthat-min 20`, `15`, `10`, same seeds: the backgrounds are the same). For O+O and Ne+Ne, use `OO_5360_jet.xml` / `NeNe_5360_jet.xml` with `grid_oo.yaml`, and check dN_ch/dη and v₂{2} against the LHC data first ([O+O and Ne+Ne at 5.36 TeV](https://jhputschke.github.io/js-contrib/run/extra-hard/#oo-and-nene-at-536-tev)).

## 6. Histograms and jets of a minimum-bias campaign

For figures beyond the class averages, `extra_hard_hists.py` makes one pass over a minimum-bias extras campaign and writes, per event, η, φ, φ − Ψ₂ and pT of the charged hadrons and anti-kT R = 0.4 jets without subtraction (charged and all hadrons), together with the extras' scatterings and initiators. It does this for the bulk, bulk + wakes and bulk + extras, and it keeps the sums for v₂(p_T) of charged hadrons with three estimators: ALICE's v₂{2, |Δη| > 1} in |η| < 0.8, the same with |Δη| > 1.5, and ⟨cos 2(φ − Ψ₂)⟩ relative to the jet-free background's plane (no non-flow from the extras). Two notebooks draw them with jackknife errors over the hydro backgrounds, and compare v₂(p_T) in two N_coll halves of the 0–10 % sample ("0–5 %", "5–10 %") with ALICE's v₂{2, |Δη| > 1} ([arXiv:1804.02944](https://arxiv.org/abs/1804.02944)) and with CMS's v₂{SP} up to 100 GeV ([arXiv:1702.00630](https://arxiv.org/abs/1702.00630)). `./get_hepdata.sh` fetches the data once into `../runs/hepdata/`; `v2pt.py` holds the estimators and the data readers.

```bash
./get_hepdata.sh                                              # ALICE + CMS v2(pT), once
python extra_hard_hists.py --prod /path/to/campaign --out hists.npz --workers 12
```

|                                                                                                                                                                   |                                                                                                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--prod`                                                                                                                                                          | the campaign directory (its `*_hadrons_bulk_jet.h5`)                                                                                                                                                                                                                                                                              |
| `--out`                                                                                                                                                           | the `.npz` (default `../runs/pbpb_extra_hard/data/extra_hard_hists.npz`)                                                                                                                                                                                                                                                          |
| `--workers`                                                                                                                                                       | processes (default 12); they take contiguous event ranges, about `--chunks-per-worker` (default 4) each, so the cores stay busy to the end. The output is the same as with one range per file (`--chunks-per-worker 0`). 1,000 Pb+Pb events × 40 oversamples: 302 s with 20 workers on the GB10 (10 fast and 10 efficiency cores) |
| [`v2pt.py`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/bulk/v2pt.py)                                                 | v₂(p_T) from the sums, the event-plane resolution, the HEPData tables                                                                                                                                                                                                                                                             |
| [`extra_hard_bulk_jets.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/bulk/extra_hard_bulk_jets.ipynb)           | the figures and ratios (numpy + matplotlib), and v₂(p_T) against ALICE and CMS                                                                                                                                                                                                                                                    |
| [`extra_hard_bulk_jets_hist.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/bulk/extra_hard_bulk_jets_hist.ipynb) | the same with `hist` + mplhep: a jackknife `block` axis, rebinning, a `Mean` profile                                                                                                                                                                                                                                              |

## Not yet

vₙ{4} and higher cumulants, ρ(v₂², [pT]), symmetry planes, multiplicity-based centrality, pT-differential vₙ(pT) in `bulk_obs.py` (v₂(p_T) is in `extra_hard_hists.py` and the notebooks), the experiments' ⟨ΔpT ΔpT⟩ correlator, pairs across oversamples (the fluid's vₙ without decay non-flow), event weights, and a C++ twin for the ROOT path.

## Next steps

- Make the files: [Extra hard scatterings](https://jhputschke.github.io/js-contrib/run/extra-hard/index.md).
- The extras in `HadronFileReader`: [Hadrons](https://jhputschke.github.io/js-contrib/analysis/hadrons/#the-events-other-hard-scatterings).
- Flow in the O+O with ISR production, from the oversamples: [O+O](https://jhputschke.github.io/js-contrib/analysis/oo/#10-flow-from-the-oversamples).
- Full reference: [`bulk/README.md`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/bulk/README.md) and [`docs/README_extra_hard.md`](https://github.com/jhputschke/js-contrib/blob/main/docs/README_extra_hard.md).
