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
| 02 | [Your first events in Python](pyjetscape/02_first_events.ipynb): three O+O events with the Pythonic interface, the stages of an event, its partons and hadrons; then η/s changed, with everything else kept the same | X-SCAPE | ready |
| 03 | [The production scripts](pyjetscape/03_production_scripts.ipynb): a real Au+Au job with a reused background, the user and job XML, seeds, p̂<sub>T</sub> windows and cross sections, hadronization, ROOT conversion; campaigns, monitoring and the OSDF | X-SCAPE | ready |
| **Analysis** ||||
| 04 | [The files and their provenance](pyjetscape/04_files_and_provenance.ipynb): pair, particlize, hadron and ROOT files of one job, the job record, what to check before combining files | analysis | ready |
| 05 | [Hadrons](pyjetscape/05_hadrons.ipynb): units and oversamples, dN/dη and identified spectra, the event record and bad runs, combining p̂<sub>T</sub> windows with their cross sections, the error budget | analysis | ready |
| 06 | [**The hydro pair**](pyjetscape/06_hydro_pair.ipynb): where a jet deposits its energy, the wake in the fluid (Mach cone, diffusion wake), whether the fluid keeps the energy, the freeze-out delay | analysis | ready |
| 07 | [**The hadron pair**](pyjetscape/07_hadron_pair.ipynb): `bulk_jet − bulk_bg (+ jet_frag)` with correlated sampling; the wake around the jets, its energy, spectrum and chemistry | analysis | ready |
| 08 | [Jets with FastJet](pyjetscape/08_jets_fastjet.ipynb): ρA subtraction, random cones and embedding, fake jets, Au+Au against the p+p reference, SoftDrop z<sub>g</sub>, the medium response in the jet cone | analysis | ready |
| **ROOT** ||||
| 09 | [ROOT files from Python](pyjetscape/09_root_from_python.ipynb): what `run_h5toROOT.py` writes, uproot (also streamed), the campaign's cross sections, weighted spectra; RDataFrame for PyROOT users | analysis | ready |
| 10 | [ROOT C++ macros](pyjetscape/10_root_cpp_macros.ipynb): `HadronFileReader.h` with ACLiC, normalised spectra, RDataFrame on RNTuples, anti-k<sub>T</sub> jets with FastJet in C++ | ROOT | ready |
| **Extending** ||||
| 11 | [Write an energy-loss module in Python](pyjetscape/11_python_module.ipynb): how X-SCAPE calls a module, a temperature-dependent energy loss in a brick, named in the XML, compared with LBT and MATTER + LBT; cost and pitfalls | X-SCAPE | ready |
| 12 | [Write your own in-run writer](pyjetscape/12_python_writer.ipynb): code that sees each event while it runs; the cooling and flow of O+O fireballs written to HDF5 and compared with Bjorken's picture | X-SCAPE | ready |
| **Advanced** ||||
| 13 | Bulk and flow: N<sub>ch</sub>, ⟨p<sub>T</sub>⟩, v₂(p<sub>T</sub>) against data | analysis | later |
| 14 | Machine-learned hydro: the live FNO and FastHydro against MUSIC | source build with PyTorch | later |
| 15 | 3D visualization of an event | PyVista | later |

*X-SCAPE*: the production container or the Mac wheel (or a source build). *Analysis*: any
setup, Colab included. Tutorial 01 tells you which tutorials run where you are. The advanced
tutorials (*later*) follow once the comparisons with measured data have been tested further;
until then the tutorials compare the model only with itself (jet against background, Au+Au
against p+p).

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
