# Python and machine learning

X-SCAPE is a C++ framework. js-contrib opens it to Python, not as a wrapper that runs a finished
binary, but down to the module level: Python can drive the event loop, read every intermediate
result as numpy arrays, and replace any stage of the simulation with a class of its own. This page
explains why that matters for heavy-ion physics and what you can do with it today; the
[Python section](../python/index.md) shows how.

## Why Python

**Machine learning lives in Python.** Neural networks are trained, evaluated and changed in PyTorch.
A surrogate model that is only usable after it has been exported, frozen and re-linked into C++ is
slow to iterate on. With js-contrib a PyTorch model can be a JETSCAPE module: it receives the
initial condition from the framework, predicts the medium, and Matter, LBT and iSS run on its
prediction as they would on MUSIC.

**Hydro surrogates need hydro.** A Fourier neural operator (FNO) learns the map from an initial
condition to the full space-time evolution of the medium. In 2+1D it reproduces MUSIC orders of
magnitude faster
([Phys. Rev. C 113 (2026) 014904](https://doi.org/10.48550/arXiv.2507.23598)). Training it, and
the 3+1D and jet-deposition models that follow, takes many MUSIC evolutions in a format a
training code reads directly. The productions write exactly that.

**Jets in a medium need pairs.** To learn how a jet changes the medium, a model needs the same
event twice: once without the jet and once with its deposited energy, on an identical initial
condition. Then `arr - arr_bg` is the jet's effect and nothing else. The two-stage productions
(MUSIC, GPU) and FastHydro (a fast solver written in PyTorch) both write such pairs.

**Physics modules are faster to prototype in Python.** An idea for an energy-loss kernel or a
hadronization model can be written as a Python class in an afternoon and run inside the full
event, next to the C++ modules, without recompiling X-SCAPE. Once it works, it ports to C++ line
by line, because the Python interface keeps the C++ names and argument order.

**Analysis is in numpy, awkward and FastJet.** Hadrons, partons and showers come out of an event,
or out of the production files, as numpy arrays and nested awkward arrays with four-vector
kinematics, ready for `fastjet` and histogramming.

## What works today

| you want to | use | page |
|---|---|---|
| run events from Python and read every result as numpy | `pyjetscape_core` and the Pythonic layer | [The Pythonic interface](../python/pythonic.md) |
| replace a stage (hard process, energy loss, hadronization, ...) with your own Python class | the module base classes, `@js.module` | [Writing modules in Python](../python/modules.md) |
| run a PyTorch FNO as the hydro, live, while you train it | `PyFNOHydro` | [Live PyTorch FNO hydro](../python/fno.md) |
| make large samples of jet-deposition pairs fast, on CPU, CUDA or Apple MPS | FastHydro | [FastHydro](../python/fasthydro.md) |
| produce FNO training data with the real MUSIC + jet chain on GPUs | the Au+Au jet production | [Au+Au jets (two stages)](../run/auau-jet.md) |
| analyse hadrons and jets from the production files | `HadronArray`, awkward, FastJet | [Analyse](../analysis/index.md) |
| look at the medium and the shower in 3D | the PyVista scripts | [3D visualization](../python/visualization.md) |

Python modules run in the container and with the Mac wheel as they are: no compiler is needed.
PyTorch is needed only for `PyFNOHydro` and FastHydro.

## Next steps

- [The Python layers](../python/index.md): where the bindings come from, and a first event from Python.
- [The physics](physics.md) of the chain the modules plug into.
- [GPU acceleration](gpu.md): why the MUSIC productions that make the training data are fast.
