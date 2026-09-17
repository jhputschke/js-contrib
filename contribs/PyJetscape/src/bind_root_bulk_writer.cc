/*******************************************************************************
 * bind_root_bulk_writer.cc
 *
 * Binds the X-SCAPE C++ ROOT bulk writer as a first-class Python type:
 *   - FastRootBulkWriter (hydro-only ROOT dump of MUSIC's native evolution
 *                         store, registered as "FastRootBulkWriter")
 *
 * FastRootBulkWriter is only compiled into libJetScape when X-SCAPE is built
 * with USE_ROOT, so the class is bound only under the same flag.  In-tree
 * builds inherit -DUSE_ROOT from the X-SCAPE CMake.  The module attribute
 * HAS_ROOT tells Python which case applies.
 *
 * The writer is configured from the XML (<FastRootBulkWriter> block) and
 * needs <Hydro><MUSIC><dump_hydro_only>1.  Python only adds it to a pipeline
 * and reads its state back; it has no setters.
 *
 * Python usage:
 *   from jetscape import create_module, HAS_ROOT
 *
 *   writer = create_module("FastRootBulkWriter")   # FastRootBulkWriter type
 *   jetscape.Add(writer)                           # after the hydro module
 *   jetscape.Init(); jetscape.Exec()
 *   jetscape.Finish()                              # writes + closes the file
 *   print(writer.GetOutFileName(), writer.GetNumberOfEventsWritten())
 *
 * pybind11 downcasts polymorphic return values to the most-derived registered
 * type, so a writer created from the XML task list also comes back as
 * FastRootBulkWriter from JetScape.GetTaskList().
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "JetScapeModuleBase.h"
#ifdef USE_ROOT
#include "FastRootBulkWriter.h"
#endif

namespace py = pybind11;
using namespace Jetscape;

void bind_root_bulk_writer(py::module_ &m) {

#ifdef USE_ROOT
  m.attr("HAS_ROOT") = true;

  py::class_<FastRootBulkWriter, JetScapeModuleBase,
             std::shared_ptr<FastRootBulkWriter>>(
      m, "FastRootBulkWriter",
      R"pbdoc(
        Hydro-only ROOT writer for the MUSIC evolution.

        Reads MUSIC's native in-memory store directly and never builds the
        framework bulk_info.data, so it is much faster and lighter than
        RootBulkWriter / PyBulkRootWriter.

        Registered module name: ``"FastRootBulkWriter"`` — instantiate via::

            writer = create_module("FastRootBulkWriter")

        or let the XML task list create it from a top-level
        ``<FastRootBulkWriter>`` block.

        Requirements:
            * ``<Hydro><MUSIC><dump_hydro_only>1`` and
              ``<output_evolution_to_memory>1`` (MUSIC is the only supported
              hydro).
            * Add it to the pipeline AFTER the hydro module.
            * Settings come from the ``<FastRootBulkWriter>`` XML block
              (out_file_name, grid_mode = native|grid, tau_stride, and the
              user grid for grid mode).

        The ROOT file is written and closed by ``Finish()`` (called by
        ``JetScape.Finish()``).  Read it back with
        ``jetscape.fast_root_bulk.read_fast_root_bulk()``.
      )pbdoc")

      // ── Configuration (read from the XML in Init()) ────────────────────────
      .def("GetOutFileName", &FastRootBulkWriter::GetOutFileName,
           "Return the output ROOT file name (<out_file_name>).")
      .def("GetGridMode", &FastRootBulkWriter::GetGridMode,
           "Return the grid mode: \"native\" (MUSIC grid) or \"grid\" "
           "(user grid).")
      .def("GetTauStride", &FastRootBulkWriter::GetTauStride,
           "Return the native-mode tau stride (every N-th stored step).")

      // ── Output state ───────────────────────────────────────────────────────
      .def("IsFileOpen", &FastRootBulkWriter::IsFileOpen,
           "Return True while the ROOT file is open (between the first event "
           "and Finish()).")
      .def("GetNumberOfEventsWritten",
           &FastRootBulkWriter::GetNumberOfEventsWritten,
           "Return the number of events filled into the tree so far.")

      // ── Layout of the most recently written event ─────────────────────────
      .def("GetNx", &FastRootBulkWriter::GetNx)
      .def("GetNy", &FastRootBulkWriter::GetNy)
      .def("GetNeta", &FastRootBulkWriter::GetNeta)
      .def("GetNtauWritten", &FastRootBulkWriter::GetNtauWritten,
           "Return the number of tau steps in the last written event.")
      .def("GetNFeatures", &FastRootBulkWriter::GetNFeatures,
           "Return the number of features per cell (4: energy_density, vx, "
           "vy, vz).")
      .def("GetTauMin", &FastRootBulkWriter::GetTauMin,
           "Return the tau of the first written step [fm/c].")
      .def("GetDtau", &FastRootBulkWriter::GetDtau,
           "Return the written tau step (includes tau_stride) [fm/c].")
      .def("GetTauFreezeout", &FastRootBulkWriter::GetTauFreezeout,
           "Return tau one step past the last stored MUSIC step [fm/c].")
      .def("get_event_layout",
           [](const FastRootBulkWriter &w) -> py::dict {
             py::dict d;
             d["out_file_name"]  = w.GetOutFileName();
             d["grid_mode"]      = w.GetGridMode();
             d["tau_stride"]     = w.GetTauStride();
             d["events_written"] = w.GetNumberOfEventsWritten();
             d["shape"] = py::make_tuple(w.GetNtauWritten(), w.GetNx(),
                                         w.GetNy(), w.GetNeta(),
                                         w.GetNFeatures());
             d["tau_min"]        = w.GetTauMin();
             d["dtau"]           = w.GetDtau();
             d["tau_freezeout"]  = w.GetTauFreezeout();
             return d;
           },
           R"pbdoc(
             Return the writer state as a dict.

             Keys
             ----
             out_file_name, grid_mode, tau_stride, events_written,
             shape (ntau, nx, ny, neta, nFeatures) of the last written event,
             tau_min, dtau, tau_freezeout.
           )pbdoc");
#else
  m.attr("HAS_ROOT") = false;
#endif
}
