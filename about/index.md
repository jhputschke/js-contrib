# About js-contrib

js-contrib is the contrib package of the [JETSCAPE](https://github.com/JETSCAPE/JETSCAPE) / [X-SCAPE](https://github.com/JETSCAPE/X-SCAPE) heavy-ion event generator (and of C-SCAPE, which is in development). It plays the same role for X-SCAPE as [fastjet-contrib](https://fastjet.hepforge.org/contrib/) does for FastJet. The framework and its physics modules stay in X-SCAPE. js-contrib adds what is needed to run them at scale and to use the results: GPU hydrodynamics, Python bindings and a Python layer, machine-learning hydro modules, HDF5 writers, the production scripts, containers and a Mac wheel, an analysis environment, and data sharing on the Open Science Data Federation (OSDF).

The physics is the standard X-SCAPE multistage chain: a 3D Monte Carlo Glauber initial state with dynamical strings, PYTHIA 8 hard scatterings, MATTER + LBT jet energy loss with recoils, energy–momentum deposition of the jet into the fluid, 3+1D viscous MUSIC hydrodynamics, Cooper–Frye particlization with iSS and string hadronization of the jets. js-contrib's productions run this chain on the GPU for large event samples: Au+Au with a jet and its medium response, O+O with initial-state radiation, and Pb+Pb Z+jet.

This section explains why the package exists, what physics the productions contain, how the GPU and Python parts fit in, and what to cite. To get going right away, see [Install](https://jhputschke.github.io/js-contrib/install/index.md) and [Run](https://jhputschke.github.io/js-contrib/run/index.md).

- **Why js-contrib**

  ______________________________________________________________________

  What it adds on top of X-SCAPE: large GPU productions with jets and their wake, ML training data, Python access to the framework, portable setups and open data.

  [Why js-contrib](https://jhputschke.github.io/js-contrib/about/why/index.md)

- **The physics**

  ______________________________________________________________________

  The event chain as the productions configure it, stage by stage, from the initial state to the hadrons, with the parameters and the publications behind each module.

  [The physics](https://jhputschke.github.io/js-contrib/about/physics/index.md)

- **GPU acceleration**

  ______________________________________________________________________

  MUSIC on CUDA and Metal: what runs on the GPU, how close it is to CPU MUSIC, and the throughput per machine.

  [GPU acceleration](https://jhputschke.github.io/js-contrib/about/gpu/index.md)

- **Python and machine learning**

  ______________________________________________________________________

  The framework in Python, modules written in Python, and Fourier neural operator (FNO) hydro with PyTorch.

  [Python and machine learning](https://jhputschke.github.io/js-contrib/about/python-ml/index.md)

- **Publications and citing**

  ______________________________________________________________________

  The papers behind the framework and every module, and how to cite js-contrib.

  [Publications and citing](https://jhputschke.github.io/js-contrib/about/references/index.md)
