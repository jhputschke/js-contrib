# Tutorials

The other sections of this site are a reference, organized by topic. The tutorials are a
path through it instead: step-by-step Jupyter notebooks that start with nothing installed
and end with your own analysis and your own modules. Each tutorial:

- states its goal, its time, what it needs, and its data and their size;
- explains the physics behind each step, linked to [The physics](../about/physics.md);
- has numbered steps you run yourself;
- ends with a **check your result** cell, which compares against the expected numbers, and
  with exercises and links to the reference pages.

The notebooks run on your own computer: in the production container, with the xscape wheel
on a Mac, or in the analysis environment. The analysis tutorials also run in Google Colab.
The pages here show them with their outputs. To run them, see [Setting up](setup.md).

The outputs shown were produced on an Apple Silicon Mac, in the environment the xscape wheel
0.2.0 is built in, with the hydro on the GPU through Metal. A notebook that needed something
else says so at its top. Where a tutorial reads fixed example files, your numbers should
match the ones shown. Where it runs X-SCAPE, they can differ slightly from setup to setup.

!!! info "In progress"
    The tutorials are being written. The list below shows which ones are ready.

## PyJetscape

| # | tutorial | needs | status |
|---|---|---|---|
| **Set up and run** ||||
| 01 | [Check your setup](pyjetscape/01_check_setup.ipynb): what your setup can do, the data, a first file and plot | any setup | ready |
| 02 | Your first events in Python: three O+O events with the Pythonic interface; change one parameter and see what it does | X-SCAPE | planned |
| 03 | The production scripts: a background with its jets, the user XML, seeds, p̂<sub>T</sub> windows, hadronization, ROOT conversion, provenance; a mini campaign, then scaling up | X-SCAPE | planned |
| **Analysis** ||||
| 04 | The files and their provenance: pair, particlize and hadron files, the job record, what to check before combining files | analysis | planned |
| 05 | Hadrons: spectra and dN/dη, oversamples and events, cross-section weights over p̂<sub>T</sub> windows | analysis | planned |
| 06 | **The hydro pair**: the jet and background fluids, Δε, the wake (Mach cone, diffusion wake), deposited against lost energy | analysis | planned |
| 07 | **The hadron pair**: jet-induced hadrons, `bulk_jet − bulk_bg + jet_frag`, the depletion behind the jet, Δφ and ΔR profiles | analysis | planned |
| 08 | Jets with FastJet: ρA subtraction, δp<sub>T</sub> from embedding, the ratio to a pp reference, SoftDrop | analysis | planned |
| **ROOT** ||||
| 09 | ROOT files from Python: what `run_h5toROOT.py` writes, uproot, RDataFrame | analysis | planned |
| 10 | ROOT C++ macros: `HadronFileReader.h`, ACLiC, FastJet in C++ | ROOT | planned |
| **Extending** ||||
| 11 | Write an energy-loss module in Python, and check it against the C++ module | X-SCAPE | planned |
| 12 | Write your own in-run writer | X-SCAPE | planned |
| **Advanced** ||||
| 13 | Bulk and flow: N<sub>ch</sub>, ⟨p<sub>T</sub>⟩, v₂(p<sub>T</sub>) against data | analysis | planned |
| 14 | Machine-learned hydro: the live FNO and FastHydro against MUSIC | source build with PyTorch | planned |
| 15 | 3D visualization of an event | PyVista | planned |

*X-SCAPE*: the production container or the Mac wheel (or a source build). *Analysis*: any
setup, Colab included. Tutorial 01 tells you which tutorials run where you are.

## The data

The tutorials read example files from the OSDF, over HTTPS, with no account needed. The
files are streamed as they are read, or downloaded once with `get_data.py`; see
[Setting up](setup.md#3-the-data). The data sets and their contents are listed in
[`datasets.json`](https://github.com/jhputschke/js-contrib/blob/main/tutorials/pyjetscape/datasets.json).

## Other contribs

The tutorials live in
[`tutorials/<contrib>/`](https://github.com/jhputschke/js-contrib/tree/main/tutorials) in
the repository, one folder per contrib. Tutorials for FastHydro and the visualization can
follow the same layout.
