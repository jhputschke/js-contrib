/*******************************************************************************
 * bind_music.cc
 *
 * Binds concrete C++ modules as first-class Python types:
 *   - MpiMusic      (MUSIC viscous hydrodynamics, registered as "MUSIC")
 *   - TrentoInitial (TRENTo initial conditions, registered as "TrentoInitial")
 *
 * Both classes are subclasses of already-bound base classes (FluidDynamics and
 * InitialState respectively), so pybind11 exposes the full method resolution
 * order to Python.  The concrete types MUST be registered IN THE SAME MODULE
 * PASS as their bases (which is the case here — all bind_*() functions are
 * called from pyjetscape_core.cc within a single PYBIND11_MODULE block).
 *
 * Python usage:
 *   from jetscape import create_module, MpiMusic, TrentoInitial
 *
 *   hydro = create_module("MUSIC")        # returns MpiMusic typed object
 *   hydro.set_preserve_bulk_info(True)    # keep bulk_info after Clear()
 *   ...
 *   bulk = hydro.get_bulk_info()
 *
 *   ini = create_module("TrentoInitial")  # returns TrentoInitial typed object
 *   ...
 *   s   = ini.get_entropy_density_numpy()
 *   info = ini.get_event_info()           # dict with EventInfo fields
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FluidDynamics.h"
#include "InitialState.h"
#include "MusicWrapper.h"
#include "TrentoInitial.h"

namespace py = pybind11;
using namespace Jetscape;

void bind_music(py::module_ &m) {

  // ── MpiMusic ────────────────────────────────────────────────────────────────
  // Concrete MUSIC viscous-hydro wrapper.  Registered as "MUSIC" in the module
  // factory.  Inherits FluidDynamics in the Python type hierarchy so that all
  // FluidDynamics methods (get_bulk_info, find_freezeout_surface, etc.) are
  // accessible on instances.
  //
  // Key difference from the Python-trampoline FluidDynamics path:
  //   * MpiMusic is a pure C++ class — it does NOT go through PyFluidDynamics.
  //   * preserve_bulk_info_ is implemented natively in MusicWrapper.h/.cc.
  //   * The downcast in create_module() returns std::shared_ptr<MpiMusic> so
  //     that pybind11 resolves the Python type as MpiMusic (not FluidDynamics).
  py::class_<MpiMusic, FluidDynamics, std::shared_ptr<MpiMusic>>(
      m, "MpiMusic",
      R"pbdoc(
        MUSIC viscous relativistic hydrodynamics module.

        Registered module name: ``"MUSIC"`` — instantiate via::

            hydro = create_module("MUSIC")

        Important setup before calling Init():
            * The JetScape XML must have ``<Hydro><MUSIC>`` settings with
              ``output_evolution_to_memory: 1`` for bulk_info to be populated.
            * Call ``hydro.set_preserve_bulk_info(True)`` so that bulk_info
              survives the ClearTasks() call at the end of each event.

        After Exec() completes, ``hydro.get_bulk_info()`` returns an
        EvolutionHistory with all stored fluid cells.
      )pbdoc")

      // ── Preserve flag ──────────────────────────────────────────────────────
      .def("set_preserve_bulk_info", &MpiMusic::set_preserve_bulk_info,
           "When True, bulk_info.data is retained after Clear() so that it "
           "can be inspected from Python after the event finishes.  Set this "
           "BEFORE calling Init().",
           py::arg("preserve"))
      .def("get_preserve_bulk_info", &MpiMusic::get_preserve_bulk_info,
           "Return whether bulk_info preservation is enabled.")

      // ── Freeze-out temperature accessor ───────────────────────────────────
      .def("GetHydroFreezeOutTemperature",
           &MpiMusic::GetHydroFreezeOutTemperature,
           "Return the freeze-out temperature set in the XML [GeV].");

  // ── TrentoInitial ───────────────────────────────────────────────────────────
  // Concrete TRENTo initial-state module.  Registered as "TrentoInitial".
  // Inherits InitialState in Python, so get_entropy_density_numpy(),
  // GetXStep(), GetYStep(), etc. are all accessible.
  //
  // The public `info_` member (EventInfo struct) is exposed as both a property
  // dict and individual named accessors for convenience.
  py::class_<TrentoInitial, InitialState, std::shared_ptr<TrentoInitial>>(
      m, "TrentoInitial",
      R"pbdoc(
        TRENTo initial-condition module.

        Registered module name: ``"TrentoInitial"`` — instantiate via::

            ini = create_module("TrentoInitial")

        After Exec() completes:
            * ``ini.get_entropy_density_numpy()`` returns the entropy-density
              grid as a 2D numpy array.
            * ``ini.get_event_info()`` returns a dict of per-event geometric
              quantities (impact parameter, Npart, Ncoll, eccentricities …).
      )pbdoc")

      // ── Centrality ────────────────────────────────────────────────────────
      .def("GetEventCentrality", &TrentoInitial::GetEventCentrality,
           "Return the event centrality percentile [0–100].")

      // ── Per-event geometry (EventInfo) ───────────────────────────────────
      .def("get_impact_parameter",
           [](const TrentoInitial &t) { return t.info_.impact_parameter; },
           "Return the impact parameter b [fm].")
      .def("get_num_participant",
           [](const TrentoInitial &t) { return t.info_.num_participant; },
           "Return the number of wounded nucleons (Npart).")
      .def("get_num_binary_collisions",
           [](const TrentoInitial &t) { return t.info_.num_binary_collisions; },
           "Return the number of binary collisions (Ncoll).")
      .def("get_total_entropy",
           [](const TrentoInitial &t) { return t.info_.total_entropy; },
           "Return the total entropy of the event.")
      .def("get_event_centrality",
           [](const TrentoInitial &t) { return t.info_.event_centrality; },
           "Return the event centrality percentile [0–100].")
      .def("get_eccentricity",
           [](const TrentoInitial &t, int order) -> double {
             auto it = t.info_.ecc.find(order);
             if (it == t.info_.ecc.end())
               throw py::key_error("eccentricity order " +
                                   std::to_string(order) + " not available");
             return it->second;
           },
           "Return the n-th order participant-plane eccentricity epsilon_n.",
           py::arg("order"))
      .def("get_participant_plane_angle",
           [](const TrentoInitial &t, int order) -> double {
             auto it = t.info_.psi.find(order);
             if (it == t.info_.psi.end())
               throw py::key_error("participant plane angle order " +
                                   std::to_string(order) + " not available");
             return it->second;
           },
           "Return the n-th order participant-plane angle psi_n [rad].",
           py::arg("order"))

      // ── Convenience: full EventInfo as a Python dict ──────────────────────
      .def("get_event_info",
           [](const TrentoInitial &t) -> py::dict {
             py::dict d;
             d["impact_parameter"]      = t.info_.impact_parameter;
             d["num_participant"]        = t.info_.num_participant;
             d["num_binary_collisions"]  = t.info_.num_binary_collisions;
             d["total_entropy"]          = t.info_.total_entropy;
             d["normalization"]          = t.info_.normalization;
             d["event_centrality"]       = t.info_.event_centrality;
             d["xmid"]                   = t.info_.xmid;
             d["ymid"]                   = t.info_.ymid;

             py::dict ecc_d, psi_d;
             for (auto &kv : t.info_.ecc)
               ecc_d[py::int_(kv.first)] = kv.second;
             for (auto &kv : t.info_.psi)
               psi_d[py::int_(kv.first)] = kv.second;
             d["eccentricities"]         = ecc_d;
             d["participant_planes"]     = psi_d;
             return d;
           },
           R"pbdoc(
             Return per-event geometric information as a Python dict.

             Keys
             ----
             impact_parameter, num_participant, num_binary_collisions,
             total_entropy, normalization, event_centrality, xmid, ymid,
             eccentricities (dict {order: epsilon_n}),
             participant_planes (dict {order: psi_n}).
           )pbdoc");
}
