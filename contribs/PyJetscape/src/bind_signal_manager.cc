/*******************************************************************************
 * Copyright (c) The JETSCAPE Collaboration, 2018
 *
 * Modular, task-based framework for simulating all aspects of heavy-ion collisions
 *
 * For the list of contributors see AUTHORS.
 *
 * Report issues at https://github.com/JETSCAPE/JETSCAPE/issues
 *
 * or via email to bugs.jetscape@gmail.com
 *
 * Distributed under the GNU General Public License 3.0 (GPLv3 or later).
 * See COPYING for details.
 ******************************************************************************/

/*******************************************************************************
 * bind_signal_manager.cc
 *
 * Binds:
 *   - JetScapeSignalManager  (singleton — provides global access to all
 *                             registered top-level physics modules)
 *
 * Key methods exposed to Python:
 *   Instance()                  → JetScapeSignalManager*  (singleton)
 *   GetHydroPointer()           → shared_ptr<FluidDynamics>           | None
 *   GetInitialStatePointer()    → shared_ptr<InitialState>            | None
 *   GetPreEquilibriumPointer()  → shared_ptr<PreequilibriumDynamics>  | None
 *   SetHydroPointer(hydro)      → register a FluidDynamics module
 *   SetInitialStatePointer(ini) → register an InitialState module
 *   SetPreEquilibriumPointer(p) → register a PreequilibriumDynamics module
 *
 * Typical use inside PyBulkRootWriter.Exec():
 *   sm    = JetScapeSignalManager.Instance()
 *   hydro = sm.GetHydroPointer()    # FluidDynamics or None
 *   if hydro is not None:
 *       bulk_info = hydro.get_bulk_info()
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "JetScapeSignalManager.h"
#include "FluidDynamics.h"
#include "InitialState.h"
#include "PreequilibriumDynamics.h"

namespace py = pybind11;
using namespace Jetscape;

void bind_signal_manager(py::module_ &m) {

  py::class_<JetScapeSignalManager>(m, "JetScapeSignalManager",
      R"pbdoc(
        Singleton signal/pointer manager for the JETSCAPE framework.

        Provides global read access to all registered top-level physics
        modules so that any module (including Python modules added to the
        pipeline via JetScape.Add()) can look up its peers by type without
        holding an explicit reference.

        The singleton is populated automatically by JetScape::Init() for all
        modules registered with JetScape.Add().  Python code that creates a
        custom module (e.g. PyBulkRootWriter) can therefore access any
        framework module from inside its Exec() method:

            from pyjetscape import JetScapeSignalManager
            sm    = JetScapeSignalManager.Instance()
            hydro = sm.GetHydroPointer()   # FluidDynamics or None
            if hydro is not None:
                info = hydro.get_bulk_info()

        Notes
        -----
        * The object returned by Instance() is owned by the C++ singleton; do
          not store it across events or after Finish().
        * All Get*Pointer() methods lock the stored weak_ptr and return the
          shared_ptr.  They return None if the module was never registered.
      )pbdoc")

      // ── Singleton accessor ─────────────────────────────────────────────────
      .def_static("Instance",
          []() -> JetScapeSignalManager * {
            return JetScapeSignalManager::Instance();
          },
          py::return_value_policy::reference,
          "Return the singleton JetScapeSignalManager instance.")

      // ── Module pointer getters ─────────────────────────────────────────────
      // All getters lock the stored weak_ptr.  They return None (Python null)
      // when the module was never registered or the lifetime has ended.

      .def("GetHydroPointer",
          [](JetScapeSignalManager &sm) -> std::shared_ptr<FluidDynamics> {
            return sm.GetHydroPointer().lock();
          },
          "Return the registered FluidDynamics module, or None if not set.")

      .def("GetInitialStatePointer",
          [](JetScapeSignalManager &sm) -> std::shared_ptr<InitialState> {
            return sm.GetInitialStatePointer().lock();
          },
          "Return the registered InitialState module, or None if not set.")

      .def("GetPreEquilibriumPointer",
          [](JetScapeSignalManager &sm)
              -> std::shared_ptr<PreequilibriumDynamics> {
            return sm.GetPreEquilibriumPointer().lock();
          },
          "Return the registered PreequilibriumDynamics module, or None if not set.")

      // ── Module pointer setters ─────────────────────────────────────────────
      // These mirror the C++ API.  In normal use, JetScape::Init() calls them
      // automatically.  They are exposed here for completeness and for unit
      // tests that build a pipeline without a full JetScape controller.

      .def("SetHydroPointer",
          [](JetScapeSignalManager &sm,
             std::shared_ptr<FluidDynamics> hydro) {
            sm.SetHydroPointer(hydro);
          },
          py::arg("hydro"),
          "Register a FluidDynamics module with the signal manager.")

      .def("SetInitialStatePointer",
          [](JetScapeSignalManager &sm,
             std::shared_ptr<InitialState> ini) {
            sm.SetInitialStatePointer(ini);
          },
          py::arg("ini"),
          "Register an InitialState module with the signal manager.")

      .def("SetPreEquilibriumPointer",
          [](JetScapeSignalManager &sm,
             std::shared_ptr<PreequilibriumDynamics> preeq) {
            sm.SetPreEquilibriumPointer(preeq);
          },
          py::arg("preeq"),
          "Register a PreequilibriumDynamics module with the signal manager.")

      // ── Convenience counters ───────────────────────────────────────────────
      .def("GetNumberOfJetSignals",
          &JetScapeSignalManager::GetNumberOfJetSignals,
          "Return the number of connected jet signals.")

      .def("GetNumberOfEdensitySignals",
          &JetScapeSignalManager::GetNumberOfEdensitySignals,
          "Return the number of connected energy-density signals.")

      .def("GetNumberOfGetHydroCellSignals",
          &JetScapeSignalManager::GetNumberOfGetHydroCellSignals,
          "Return the number of connected GetHydroCell signals.");
}
