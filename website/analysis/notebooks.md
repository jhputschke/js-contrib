# Notebooks

Every notebook sits next to the code or the production it belongs to; this page is the index.
Each one reads its inputs from environment variables (the defaults are in its first code cell),
so the same notebook runs on the shipped example files, on your own run or on a downloaded
campaign. Most are committed with their outputs, so they can be read on GitHub before running
anything.

## Running a notebook

Set the notebook's variables, then open it in Jupyter or run it headless:

```bash
source ~/.venvs/js_analysis/bin/activate                  # the analysis venv
HADRON_DIR=/path/to/campaign jupyter lab contribs/PyJetscape/example/analysis/tutorials/hadron_distributions.ipynb
HADRON_DIR=/path/to/campaign jupyter nbconvert --to notebook --execute --inplace \
    contribs/PyJetscape/example/analysis/tutorials/hadron_distributions.ipynb
```

The notebooks name the kernel they were saved with; in the venv choose `Python 3 (ipykernel)`, or
the kernel registered with `setup_analysis_env.sh --kernel NAME` (for VS Code or another
JupyterLab). On the Mac, the xscape environment runs them too, from your copy of the examples
([Mac (xscape wheel)](../install/mac.md)).

**Environments** in the tables: *venv* = the [analysis environment](../install/analysis-env.md)
(no X-SCAPE, no ROOT); *venv + ROOT* = the same with ROOT ≥ 6.34 ([ROOT and C++](root.md#1-set-up));
*X-SCAPE* = an environment that runs X-SCAPE: the [Mac wheel](../install/mac.md) or a
[source build](../install/source.md).

## Reading the files: tutorials

| notebook | what it shows | input | environment |
|---|---|---|---|
| [`analysis/tutorials/hadron_distributions.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/tutorials/hadron_distributions.ipynb) | η, φ, pT of background, background + wake, jet fragments and the wake, in the lab and relative to the jet; checks how the oversamples were drawn | `HADRON_DIR` (hadron files of `hadronize.py` or of hadrons-only jobs), `MAX_EVENTS`, `PAIRED` (`auto`, `1`, `0`) | venv |
| [`analysis/tutorials/jet_fastjet.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/tutorials/jet_fastjet.ipynb) | FastJet's classic interface: jet spectrum and fragmentation function of the jet hadrons, jet areas, ρ from kT jets, random cones, the ρA-subtracted spectrum, embedding δpT | `HADRON_DIR`, `MAX_EVENTS`, `MAX_OVERSAMPLES`, `PARTICLES` (`charged`, `full`) | venv |
| [`analysis/tutorials/jet_fastjet_awkward.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/tutorials/jet_fastjet_awkward.ipynb) | the same with the array interface, every sample clustered in one call; exclusive jets, SoftDrop, Lund planes, energy correlators, N-subjettiness; areas with hand-made ghosts | as above | venv |

## Hydro and the wake

| notebook | what it shows | input | environment |
|---|---|---|---|
| [`prod_AuAu_0_10_jet/jet_wake.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/prod_AuAu_0_10_jet/jet_wake.ipynb) | one pair from the energy density through the deposit and freeze-out to hadrons, and the jet energy balance ([walk-through](hydro-hadrons.md#5-one-event-end-to-end-jet_wakeipynb)) | `WAKE_H5`: a pair file, with its particlize and hadron files next to it; `MUSIC_EOS_TABLE` | venv |
| [`analysis/wake_observables.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/wake_observables.ipynb) | jet energy loss and the wake at parton and hydro level over a campaign | `WAKE_OBS`: `wake_observables.h5` of `wake_observables.py` | venv |
| [`analysis/wake_hadrons.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/wake_hadrons.ipynb) | the same at hadron level: jet frame, ΔR profiles, identified spectra, energy balance | `WAKE_HAD`: `wake_hadrons.h5` of `wake_hadrons.py`, and `WAKE_OBS` | venv |
| [`hydro_hist_vs_surface/hydro_hist_vs_surface.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/hydro_hist_vs_surface/hydro_hist_vs_surface.ipynb) | hadrons from a surface found in the stored hydro history against MUSIC's full surface: yields, ⟨pT⟩, v_n, the wake, corrections ([summary](hydro-hadrons.md#6-hadrons-from-the-stored-hydro-history)) | `HVS_SUMMARY`: the study's `summary.npz` | venv (the study itself: X-SCAPE) |
| [`prod_AuAu_0_10/check_output.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/prod_AuAu_0_10/check_output.ipynb) | hydro-only files: a sanity scan (finite, e ≥ 0, \|v\| < 1), e and flow against τ, freeze-out times | `PROD_H5_GLOB` | venv |

## Z+jet

| notebook | what it shows | input | environment |
|---|---|---|---|
| [`analysis/zjet/zjet_wake.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis/zjet/zjet_wake.ipynb) | the Z–hadron wake (paired truth) in Δφ and Δη, Pb+Pb − pp, the wake against pT,Z, one event's hydro wake and movie frames | `ZJET_HADRONS` (directory or glob), optional `ZJET_VACUUM`, `ZJET_PAIR`, `ZJET_PAIR_EVENT`, `ZJET_MOVIE` | venv |

The campaign analyses themselves are scripts, not notebooks: [Jets with FastJet](jets.md#7-worked-examples-the-zjet-campaign-analyses).

## O+O

| notebook | what it shows | input | environment |
|---|---|---|---|
| [`prod_OO_isr/oo_isr_overview.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/prod_OO_isr/oo_isr_overview.ipynb) | one job: the medium at three τ with the freeze-out contour and the hard scattering; η, φ, pT and Δφ to the leading parton of bulk and jet hadrons | `OO_STEM`; `XSCAPE_BUILD` for the EoS | venv |
| [`prod_OO_isr/hard_sub_compare/hard_sub_compare.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/prod_OO_isr/hard_sub_compare/hard_sub_compare.ipynb) | two campaigns with and without the hard-momentum subtraction: the centrality bias of a multiplicity estimator, dN/dη, dE/dη and spectra, per window; N_coll-weighted | `HSC_DIR` (the output of `run_compare.sh`) | venv |
| [`pythonic/oo_analysis.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/pythonic/oo_analysis.ipynb) | the O+O hadron files of a public test run, streamed from the OSDF: the job, the event record, nested awkward views, spectra, jets with FastJet | the OSDF (`osdf:///fno4hic/test/OO_test`), or `OO_DIR` for downloaded files | venv (with `pelicanfs`) |
| [`pythonic/oo_workflow.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/pythonic/oo_workflow.ipynb) | three O+O events through the full chain with the Pythonic interface, the hadron files written and read back, awkward views, FastJet on every oversample in one call, a jet embedded in the bulk | runs X-SCAPE; writes `OO_OUT` (default `out_oo`) | X-SCAPE |

More on the Pythonic notebooks: [The Pythonic interface](../python/pythonic.md).

## ROOT

| notebook | what it shows | input | environment |
|---|---|---|---|
| [`analysis_root/read_hadrons.ipynb`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/analysis_root/read_hadrons.ipynb) | `read_hadrons.C` one step per cell: the C++ reader by example, pT spectra and Δφ per source and for the wake, drawn | `HADRON_DIR` (a campaign converted by `run_h5toROOT.py`), `HADRON_OUT`, `MAX_SAMPLES` | venv + ROOT, ROOT's C++ kernel (`pip install metakernel`) |

## FastHydro

FastHydro's notebooks (`jet_wake`, `hadron_wake`, `bulk_vs_music`) are in
[`contribs/FastHydro/notebooks/`](https://github.com/jhputschke/js-contrib/tree/main/contribs/FastHydro/notebooks);
see [FastHydro](../python/fasthydro.md).

## Next steps

- Before running a notebook on a campaign, check its settings: [What a production writes](files.md#6-settings-to-check-before-combining-files).
- Get example data: [Data on the OSDF](../run/osdf.md).
