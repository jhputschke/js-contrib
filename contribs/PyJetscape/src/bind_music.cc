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
 *   # with <dump_hydro_only>1: read MUSIC's native store (no bulk_info.data)
 *   evo = hydro.get_native_evolution_numpy()   # (ntau, nx, ny, neta, 4)
 *
 *   ini = create_module("TrentoInitial")  # returns TrentoInitial typed object
 *   ...
 *   s   = ini.get_entropy_density_numpy()
 *   info = ini.get_event_info()           # dict with EventInfo fields
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "FluidCellInfo.h"
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
           "Return the freeze-out temperature set in the XML [GeV].")

      // ── Hydro-only dump: MUSIC's native in-memory store ────────────────────
      // With <Hydro><MUSIC><dump_hydro_only>1 MUSIC keeps its own evolution
      // store and never fills bulk_info.data (only the grid metadata).  These
      // read that store directly, the same way FastRootBulkWriter does.  Both
      // flags are read from the XML in InitializeHydro(), so only getters are
      // exposed.
      .def("get_dump_hydro_only", &MpiMusic::get_dump_hydro_only,
           "Return whether <dump_hydro_only> is enabled (native store kept, "
           "bulk_info.data not built).")
      .def("get_skip_surface", &MpiMusic::get_skip_surface,
           "Return whether <skip_surface> is enabled (freeze-out surface not "
           "exported to the framework).")
      .def("get_number_of_fluid_cells", &MpiMusic::get_number_of_fluid_cells,
           "Return the number of cells in MUSIC's native evolution store "
           "(0 before InitializeHydro() or after the store was released).")
      .def("clear_hydro_info_from_memory",
           &MpiMusic::clear_hydro_info_from_memory,
           "Release MUSIC's native evolution store.  FastRootBulkWriter does "
           "this after writing each event.")
      .def("get_native_fluid_cell",
           [](MpiMusic &h, int idx) {
             const int n = h.get_number_of_fluid_cells();
             if (idx < 0 || idx >= n)
               throw py::index_error("native cell index " +
                                     std::to_string(idx) + " out of range [0, " +
                                     std::to_string(n) + ")");
             FluidCellInfo cell;
             h.get_native_fluid_cell(idx, cell);
             return cell;
           },
           R"pbdoc(
             Return one cell of MUSIC's native store as a FluidCellInfo.

             ``idx`` is the flat tau-major (tau, x, y, eta) index, i.e.
             ``((it*nx + ix)*ny + iy)*neta + ieta`` with the grid sizes from
             ``get_bulk_info()``.
           )pbdoc",
           py::arg("idx"))
      .def("get_native_evolution_numpy",
           [](MpiMusic &h, int tau_stride) -> py::array_t<float> {
             if (tau_stride < 1)
               throw std::invalid_argument(
                   "get_native_evolution_numpy: tau_stride must be >= 1, got " +
                   std::to_string(tau_stride));
             const auto &g = h.get_bulk_info();
             const long n_per_step = (long)g.nx * g.ny * g.neta;
             const int num_cells = h.get_number_of_fluid_cells();
             if (num_cells <= 0 || n_per_step <= 0)
               throw std::runtime_error(
                   "get_native_evolution_numpy: MUSIC native store is empty. "
                   "Needs <dump_hydro_only>1 and output_evolution_to_memory=1, "
                   "and must be called before FastRootBulkWriter releases the "
                   "store for this event.");

             // Same tau thinning as FastRootBulkWriter native mode.
             const int ntau_native = (int)(num_cells / n_per_step);
             const int ntau_out = (ntau_native + tau_stride - 1) / tau_stride;
             py::array_t<float> arr(std::vector<py::ssize_t>{
                 ntau_out, g.nx, g.ny, g.neta, 4});
             float *out = arr.mutable_data();
             {
               py::gil_scoped_release release;
               FluidCellInfo cell;
               for (int it = 0; it < ntau_native; it += tau_stride) {
                 const long base = (long)it * n_per_step;
                 for (long ic = 0; ic < n_per_step; ic++) {
                   h.get_native_fluid_cell((int)(base + ic), cell);
                   *out++ = (float)cell.energy_density;
                   *out++ = (float)cell.vx;
                   *out++ = (float)cell.vy;
                   *out++ = (float)cell.vz;
                 }
               }
             }
             return arr;
           },
           py::arg("tau_stride") = 1,
           R"pbdoc(
             Copy MUSIC's native evolution store into a numpy array in one C++
             pass (GIL released), without building bulk_info.data.

             The values and layout are exactly what FastRootBulkWriter writes
             in ``native`` mode, so ``arr.ravel()`` equals one ``user_res``
             entry of its ROOT output.

             Requires ``<Hydro><MUSIC><dump_hydro_only>1``.  If a
             FastRootBulkWriter is in the pipeline it releases the store at the
             end of its Exec(), so read the store before that runs.

             Parameters
             ----------
             tau_stride : int, default 1
                 Keep every N-th stored tau step.

             Returns
             -------
             np.ndarray, shape (ntau, nx, ny, neta, 4), dtype float32
                 Features: [energy_density, vx, vy, vz].  Grid spacing and
                 origin are in ``get_bulk_info()`` (dtau is multiplied by
                 ``tau_stride``).
           )pbdoc");

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
