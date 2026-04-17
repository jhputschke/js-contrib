/*******************************************************************************
 * bind_evolution.cc
 *
 * Binds:
 *   - FluidCellInfo   (all public fields)
 *   - SurfaceCellInfo (all public fields)
 *   - EvolutionHistory (grid metadata, zero-copy data_vector numpy view,
 *                       FluidCellInfo vector, FromVector)
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "FluidCellInfo.h"
#include "FluidEvolutionHistory.h"
#include "SurfaceCellInfo.h"

namespace py = pybind11;
using namespace Jetscape;

void bind_evolution(py::module_ &m) {

  // ── FluidCellInfo ────────────────────────────────────────────────────────────
  py::class_<FluidCellInfo>(m, "FluidCellInfo")
      .def(py::init<>())
      .def_readwrite("energy_density", &FluidCellInfo::energy_density,
                     "Local energy density [GeV/fm^3].")
      .def_readwrite("entropy_density", &FluidCellInfo::entropy_density,
                     "Local entropy density [1/fm^3].")
      .def_readwrite("temperature", &FluidCellInfo::temperature,
                     "Local temperature [GeV].")
      .def_readwrite("pressure", &FluidCellInfo::pressure,
                     "Thermal pressure [GeV/fm^3].")
      .def_readwrite("qgp_fraction", &FluidCellInfo::qgp_fraction,
                     "QGP fraction (QGP+HRG phase).")
      .def_readwrite("mu_B", &FluidCellInfo::mu_B,
                     "Net baryon chemical potential [GeV].")
      .def_readwrite("mu_C", &FluidCellInfo::mu_C,
                     "Net charge chemical potential [GeV].")
      .def_readwrite("mu_S", &FluidCellInfo::mu_S,
                     "Net strangeness chemical potential [GeV].")
      .def_readwrite("vx", &FluidCellInfo::vx, "Flow velocity vx.")
      .def_readwrite("vy", &FluidCellInfo::vy, "Flow velocity vy.")
      .def_readwrite("vz", &FluidCellInfo::vz, "Flow velocity vz.")
      .def_readwrite("bulk_Pi", &FluidCellInfo::bulk_Pi,
                     "Bulk viscous pressure [GeV/fm^3].")
      .def("__repr__", [](const FluidCellInfo &c) {
        return "<FluidCellInfo ed=" + std::to_string(c.energy_density) +
               " T=" + std::to_string(c.temperature) + ">";
      });

  // ── SurfaceCellInfo ──────────────────────────────────────────────────────────
  py::class_<SurfaceCellInfo>(m, "SurfaceCellInfo")
      .def(py::init<>())
      .def_readwrite("tau", &SurfaceCellInfo::tau,
                     "Proper time tau [fm/c].")
      .def_readwrite("x", &SurfaceCellInfo::x, "Transverse x [fm].")
      .def_readwrite("y", &SurfaceCellInfo::y, "Transverse y [fm].")
      .def_readwrite("eta", &SurfaceCellInfo::eta, "Space-time rapidity.")
      .def_readwrite("energy_density", &SurfaceCellInfo::energy_density,
                     "Energy density on the surface [GeV/fm^3].")
      .def_readwrite("entropy_density", &SurfaceCellInfo::entropy_density,
                     "Entropy density on the surface [1/fm^3].")
      .def_readwrite("temperature", &SurfaceCellInfo::temperature,
                     "Temperature on the surface [GeV].")
      .def_readwrite("pressure", &SurfaceCellInfo::pressure,
                     "Pressure on the surface [GeV/fm^3].")
      .def_readwrite("qgp_fraction", &SurfaceCellInfo::qgp_fraction)
      .def_readwrite("mu_B", &SurfaceCellInfo::mu_B)
      .def_readwrite("mu_Q", &SurfaceCellInfo::mu_Q)
      .def_readwrite("mu_S", &SurfaceCellInfo::mu_S)
      .def_readwrite("bulk_Pi", &SurfaceCellInfo::bulk_Pi,
                     "Bulk viscous pressure [GeV/fm^3].")
      // d3sigma_mu[4] and umu[4] and pi[10] as list
      .def_property("d3sigma_mu",
          [](const SurfaceCellInfo &s) {
            return std::vector<float>(s.d3sigma_mu, s.d3sigma_mu + 4);
          },
          [](SurfaceCellInfo &s, const std::vector<float> &v) {
            for (int i = 0; i < 4 && i < (int)v.size(); ++i)
              s.d3sigma_mu[i] = v[i];
          }, "Surface normal vector d^3 sigma_mu (4-vector).")
      .def_property("umu",
          [](const SurfaceCellInfo &s) {
            return std::vector<float>(s.umu, s.umu + 4);
          },
          [](SurfaceCellInfo &s, const std::vector<float> &v) {
            for (int i = 0; i < 4 && i < (int)v.size(); ++i)
              s.umu[i] = v[i];
          }, "4-velocity u^mu.")
      .def_property("pi",
          [](const SurfaceCellInfo &s) {
            return std::vector<float>(s.pi, s.pi + 10);
          },
          [](SurfaceCellInfo &s, const std::vector<float> &v) {
            for (int i = 0; i < 10 && i < (int)v.size(); ++i)
              s.pi[i] = v[i];
          }, "Shear stress tensor pi (10 independent components).")
      .def("__repr__", [](const SurfaceCellInfo &s) {
        return "<SurfaceCellInfo tau=" + std::to_string(s.tau) +
               " T=" + std::to_string(s.temperature) + ">";
      });

  // ── EvolutionHistory ─────────────────────────────────────────────────────────
  py::class_<EvolutionHistory>(m, "EvolutionHistory",
      R"pbdoc(
        Stores the hydrodynamic evolution history on a (tau,x,y,eta) grid.

        Two storage modes:
          * ``data``        — vector of FluidCellInfo objects (used by jet quenching)
          * ``data_vector`` — flat float32 vector (used by external hydro modules)

        Grid parameters: tau_min, dtau, ntau, x_min, dx, nx, y_min, dy, ny,
                         eta_min, deta, neta, boost_invariant, tau_eta_is_tz.
      )pbdoc")
      .def(py::init<>())
      // Grid metadata — read/write from Python
      .def_readwrite("tau_min", &EvolutionHistory::tau_min)
      .def_readwrite("dtau",    &EvolutionHistory::dtau)
      .def_readwrite("x_min",   &EvolutionHistory::x_min)
      .def_readwrite("dx",      &EvolutionHistory::dx)
      .def_readwrite("y_min",   &EvolutionHistory::y_min)
      .def_readwrite("dy",      &EvolutionHistory::dy)
      .def_readwrite("eta_min", &EvolutionHistory::eta_min)
      .def_readwrite("deta",    &EvolutionHistory::deta)
      .def_readwrite("ntau",    &EvolutionHistory::ntau)
      .def_readwrite("nx",      &EvolutionHistory::nx)
      .def_readwrite("ny",      &EvolutionHistory::ny)
      .def_readwrite("neta",    &EvolutionHistory::neta)
      .def_readwrite("boost_invariant", &EvolutionHistory::boost_invariant)
      .def_readwrite("tau_eta_is_tz",   &EvolutionHistory::tau_eta_is_tz)
      .def_readwrite("data_info",       &EvolutionHistory::data_info)
      // Coordinate helpers
      .def("Tau0",   &EvolutionHistory::Tau0)
      .def("TauMax", &EvolutionHistory::TauMax)
      .def("XMin",   &EvolutionHistory::XMin)
      .def("XMax",   &EvolutionHistory::XMax)
      .def("YMin",   &EvolutionHistory::YMin)
      .def("YMax",   &EvolutionHistory::YMax)
      .def("EtaMin", &EvolutionHistory::EtaMin)
      .def("EtaMax", &EvolutionHistory::EtaMax)
      .def("get_data_size", &EvolutionHistory::get_data_size,
           "Return len(data) — number of FluidCellInfo entries.")
      // Zero-copy numpy view of data_vector (float32 raw buffer)
      .def("data_vector_numpy",
           [](EvolutionHistory &h) -> py::array_t<float> {
             if (h.data_vector.empty())
               return py::array_t<float>({0}, {sizeof(float)}, nullptr);
             return py::array_t<float>(
                 {(py::ssize_t)h.data_vector.size()},
                 {sizeof(float)},
                 h.data_vector.data(),
                 py::cast(&h));
           },
           py::return_value_policy::reference_internal,
           R"pbdoc(
             Zero-copy numpy float32 view of the raw data_vector buffer.
             Shape: (ntau * nx * ny * neta * n_fields,).
             The field layout is described by data_info.
             Valid as long as the EvolutionHistory object is alive.
           )pbdoc")
      // Populate data_vector from a numpy array via FromVector
      .def("FromVector",
           &EvolutionHistory::FromVector,
           R"pbdoc(
             Populate data_vector and grid metadata from a flat float32 vector.

             Parameters match EvolutionHistory::FromVector exactly.
             data_ must have length ntau * nx * ny * neta * len(data_info_).
           )pbdoc",
           py::arg("data_"), py::arg("data_info_"),
           py::arg("tau_min"), py::arg("dtau"),
           py::arg("x_min"), py::arg("dx"), py::arg("nx"),
           py::arg("y_min"), py::arg("dy"), py::arg("ny"),
           py::arg("eta_min"), py::arg("deta"), py::arg("neta"),
           py::arg("tau_eta_is_tz"))
      // Access the FluidCellInfo vector (used by jet quenching)
      .def("get_fluid_cell",
           &EvolutionHistory::GetFluidCell,
           "Return FluidCellInfo at grid index (id_tau, id_x, id_y, id_eta).",
           py::arg("id_tau"), py::arg("id_x"), py::arg("id_y"),
           py::arg("id_eta"))
      .def("cell_index",
           &EvolutionHistory::CellIndex,
           "Return flat index for grid position (id_tau, id_x, id_y, id_eta).",
           py::arg("id_tau"), py::arg("id_x"), py::arg("id_y"),
           py::arg("id_eta"))
      // Interpolated access: same interface used by BulkRootWriter::Exec()
      .def("get",
           &EvolutionHistory::get,
           R"pbdoc(
             Return interpolated FluidCellInfo at continuous space-time
             coordinates (tau, x, y, eta).

             Uses trilinear interpolation over the stored grid.  Coordinates
             outside the grid are clamped to the boundary.

             Parameters
             ----------
             tau  : float  — proper time [fm/c]
             x    : float  — transverse coordinate x [fm]
             y    : float  — transverse coordinate y [fm]
             eta  : float  — space-time rapidity (use 0 for boost-invariant)

             Returns
             -------
             FluidCellInfo with interpolated energy_density, temperature,
             vx, vy, vz, etc.
           )pbdoc",
           py::arg("tau"), py::arg("x"), py::arg("y"), py::arg("eta"))
      .def("get_at_time_step",
           &EvolutionHistory::GetAtTimeStep,
           R"pbdoc(
             Return FluidCellInfo at a fixed tau step and continuous (x, y, eta).

             Parameters
             ----------
             id_tau : int   — tau grid index
             x      : float — transverse x [fm]
             y      : float — transverse y [fm]
             eta    : float — space-time rapidity
           )pbdoc",
           py::arg("id_tau"), py::arg("x"), py::arg("y"), py::arg("eta"))
      .def("clear_up_evolution_data", &EvolutionHistory::clear_up_evolution_data,
           "Clear the FluidCellInfo data vector.")
      // Fast bulk export: runs the cell loop in C++ and returns a numpy array
      // in a single boundary crossing, avoiding ~850K per-cell pybind11 calls.
      .def("to_numpy",
           [](const EvolutionHistory &h, int n_features) -> py::array_t<float> {
             if (n_features < 1 || n_features > 6)
               throw std::invalid_argument(
                   "to_numpy: n_features must be 1-6, got " +
                   std::to_string(n_features));
             if (h.data.empty())
               throw std::runtime_error(
                   "to_numpy: bulk_info.data is empty — run EvolveHydro() first.");

             py::array_t<float> arr({h.ntau, h.nx, h.ny, n_features});
             auto buf = arr.mutable_unchecked<4>();

             for (int k = 0; k < h.ntau; ++k)
               for (int i = 0; i < h.nx; ++i)
                 for (int j = 0; j < h.ny; ++j) {
                   const auto &c = h.data[h.CellIndex(k, i, j, 0)];
                   if (n_features >= 1) buf(k, i, j, 0) = c.energy_density;
                   if (n_features >= 2) buf(k, i, j, 1) = c.temperature;
                   if (n_features >= 3) buf(k, i, j, 2) = c.vx;
                   if (n_features >= 4) buf(k, i, j, 3) = c.vy;
                   if (n_features >= 5) buf(k, i, j, 4) = c.entropy_density;
                   if (n_features >= 6) buf(k, i, j, 5) = c.pressure;
                 }
             return arr;
           },
           py::arg("n_features") = 4,
           R"pbdoc(
             Convert bulk_info.data to a numpy float32 array in a single C++
             pass, avoiding per-cell pybind11 overhead.

             Parameters
             ----------
             n_features : int (1–6, default 4)
                 Number of fields to extract per cell.
                 Layout: [energy_density, temperature, vx, vy,
                          entropy_density, pressure]

             Returns
             -------
             np.ndarray, shape (ntau, nx, ny, n_features), dtype float32
           )pbdoc");
}
