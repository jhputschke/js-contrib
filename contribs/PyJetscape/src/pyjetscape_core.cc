/*******************************************************************************
 * Copyright (c) The JETSCAPE Collaboration, 2018
 *
 * Python binding entry point for the JETSCAPE framework.
 * Compiled into pyjetscape_core.so — import as pyjetscape.pyjetscape_core
 * or indirectly via python/jetscape/__init__.py.
 ******************************************************************************/

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations for each binding compilation unit
void bind_framework(py::module_ &m);
void bind_evolution(py::module_ &m);
void bind_initial_state(py::module_ &m);
void bind_fluid_dynamics(py::module_ &m);
void bind_music(py::module_ &m);
void bind_signal_manager(py::module_ &m);

PYBIND11_MODULE(pyjetscape_core, m) {
  m.doc() = R"pbdoc(
    pyjetscape_core — Python bindings for the JETSCAPE heavy-ion simulation framework.

    Exposes C++ classes and the module factory so that Python code can:
      * Drive a full JETSCAPE simulation (JetScape, Init/Exec/Finish)
      * Instantiate registered C++ modules by name (create_module)
      * Subclass FluidDynamics in Python (PyFNOHydro trampoline)
      * Subclass JetScapeModuleBase in Python (PyBulkRootWriter and similar)
      * Access InitialState and PreequilibriumDynamics data as numpy arrays
      * Read EvolutionHistory and SurfaceCellInfo after hydro finishes
      * Query globally registered modules via JetScapeSignalManager
  )pbdoc";

  bind_framework(m);
  bind_evolution(m);
  bind_initial_state(m);
  bind_fluid_dynamics(m);
  // Concrete module bindings — must come AFTER base-class bindings above so
  // that pybind11 can resolve the inheritance chain (MpiMusic : FluidDynamics,
  // TrentoInitial : InitialState).
  bind_music(m);
  // Singleton signal manager — must come AFTER all module base-class bindings
  // (FluidDynamics, InitialState, PreequilibriumDynamics) so that the return
  // types of GetHydroPointer() etc. are already registered.
  bind_signal_manager(m);
}
