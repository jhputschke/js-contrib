/*******************************************************************************
 * bind_fluid_dynamics.cc
 *
 * Binds:
 *   - PreequilibriumDynamics  (with numpy views of all public stress-energy fields)
 *   - FluidDynamics           (with PyFluidDynamics trampoline that lets Python
 *                              subclasses override InitializeHydro + EvolveHydro)
 *
 * Key helpers on PyFluidDynamics (exposed to Python on FluidDynamics instances):
 *   get_ini_pointer()            → shared_ptr<InitialState>
 *   get_preeq_pointer()          → shared_ptr<PreequilibriumDynamics>
 *   get_bulk_info_mutable()      → EvolutionHistory&  (mutable)
 *   get_surface_cells()          → list[SurfaceCellInfo]
 *   set_hydro_grid_info(...)     → populate bulk_info grid metadata
 *   store_fluid_cells_from_numpy(arr, n_features)
 *                                → fill bulk_info.data from (n_features,nx,ny,ntau)
 *                                   numpy array
 *   set_hydro_status_finished()  → mark hydro done so GetHydroInfo() works
 *   find_freezeout_surface(T_sw) → call FindAConstantTemperatureSurface and
 *                                  store result in surfaceCellVector_
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "FluidDynamics.h"
#include "FluidEvolutionHistory.h"
#include "FluidCellInfo.h"
#include "SurfaceCellInfo.h"
#include "InitialState.h"
#include "PreequilibriumDynamics.h"
#include "JetScapeModuleBase.h"
#include "LiquefierBase.h"

namespace py = pybind11;
using namespace Jetscape;

// ── Helpers: zero-copy 1D numpy view of a std::vector<double> ───────────────
// The capsule keeps a pointer to `parent` alive (the Python object owning
// the PreequilibriumDynamics).
template <typename Parent>
static py::array_t<double> vec_to_numpy(std::vector<double> &vec,
                                         py::object parent) {
  if (vec.empty())
    return py::array_t<double>({0}, {sizeof(double)}, nullptr);
  return py::array_t<double>({(py::ssize_t)vec.size()},
                              {sizeof(double)},
                              vec.data(),
                              parent);
}

// ── PyFluidDynamics trampoline ────────────────────────────────────────────────
// Inherits FluidDynamics and overrides virtual methods via PYBIND11_OVERRIDE so
// that a Python subclass can provide InitializeHydro() and EvolveHydro().
// Also exposes protected members of FluidDynamics as public accessor methods.
class PyFluidDynamics : public FluidDynamics {
public:
  using FluidDynamics::FluidDynamics; // inherit constructors

  // When true, Clear() skips clear_up_evolution_data() so that bulk_info.data
  // survives the JetScapeTask::ClearTasks() call at the end of each event.
  // Set via set_preserve_bulk_info(True) from Python before running.
  // EvolveHydro() should call clear_up_evolution_data() explicitly at its own
  // start so multi-event runs remain correct.
  bool preserve_bulk_info_ = false;

  // ── Virtual overrides (dispatched to Python) ───────────────────────────────
  void InitializeHydro(Parameter parameter_list) override {
    PYBIND11_OVERRIDE(void, FluidDynamics, InitializeHydro, parameter_list);
  }
  void EvolveHydro() override {
    PYBIND11_OVERRIDE(void, FluidDynamics, EvolveHydro);
  }
  void Init() override {
    PYBIND11_OVERRIDE(void, FluidDynamics, Init);
  }
  void Exec() override {
    PYBIND11_OVERRIDE(void, FluidDynamics, Exec);
  }
  void Clear() override {
    if (preserve_bulk_info_) {
      // Keep bulk_info.data alive so Python can inspect it after Exec().
      // Only clear the freeze-out surface (re-computed each event).
      clearSurfaceCellVector();
      return;
    }
    PYBIND11_OVERRIDE(void, FluidDynamics, Clear);
  }

  // ── Protected-member accessors exposed to Python ───────────────────────────
  std::shared_ptr<InitialState> get_ini_pointer() { return ini; }
  std::shared_ptr<PreequilibriumDynamics> get_preeq_pointer() {
    return pre_eq_ptr;
  }
  EvolutionHistory &get_bulk_info_mutable() { return bulk_info; }
  std::vector<SurfaceCellInfo> &get_surface_cells() {
    return surfaceCellVector_;
  }

  // ── Grid metadata setter ───────────────────────────────────────────────────
  // Call this in EvolveHydro() before store_fluid_cells_from_numpy().
  void set_hydro_grid_info(float tau_min, float dtau, int ntau, float x_min,
                           float dx, int nx, float y_min, float dy, int ny,
                           float eta_min, float deta, int neta,
                           bool boost_inv = true,
                           bool tau_eta_is_tz = false) {
    bulk_info.tau_min        = tau_min;
    bulk_info.dtau           = dtau;
    bulk_info.ntau           = ntau;
    bulk_info.x_min          = x_min;
    bulk_info.dx             = dx;
    bulk_info.nx             = nx;
    bulk_info.y_min          = y_min;
    bulk_info.dy             = dy;
    bulk_info.ny             = ny;
    bulk_info.eta_min        = eta_min;
    bulk_info.deta           = deta;
    bulk_info.neta           = neta;
    bulk_info.boost_invariant = boost_inv;
    bulk_info.tau_eta_is_tz   = tau_eta_is_tz;
  }

  // ── Fluid-cell writer ──────────────────────────────────────────────────────
  // Fills bulk_info.data from a numpy array shaped (n_features, nx, ny, ntau).
  // Expected feature layout (same convention as FnoHydro):
  //   n_features == 4: [energy_density, temperature, vx, vy]
  //   n_features == 3: [energy_density, vx, vy]           (ed already un-normalised)
  //
  // The Python EvolveHydro() is responsible for any tau-normalisation of the
  // FNO output before calling this method.
  void store_fluid_cells_from_numpy(
      py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    if (buf.ndim != 4)
      throw std::runtime_error(
          "store_fluid_cells_from_numpy: expected array of shape "
          "(n_features, nx, ny, ntau)");

    int nf  = (int)buf.shape[0];
    int nx_ = (int)buf.shape[1];
    int ny_ = (int)buf.shape[2];
    int nt_ = (int)buf.shape[3];

    if (nx_ != bulk_info.nx || ny_ != bulk_info.ny || nt_ != bulk_info.ntau)
      throw std::runtime_error(
          "store_fluid_cells_from_numpy: array shape (" +
          std::to_string(nx_) + "," + std::to_string(ny_) + "," +
          std::to_string(nt_) + ") does not match bulk_info grid (" +
          std::to_string(bulk_info.nx) + "," + std::to_string(bulk_info.ny) +
          "," + std::to_string(bulk_info.ntau) + "). "
          "Call set_hydro_grid_info() first.");

    float *ptr = static_cast<float *>(buf.ptr);
    // C-contiguous layout [feat][ix][iy][itau]
    auto idx = [&](int f, int ix, int iy, int it) {
      return f * nx_ * ny_ * nt_ + ix * ny_ * nt_ + iy * nt_ + it;
    };

    bulk_info.data.clear();
    bulk_info.data.reserve((std::size_t)nt_ * nx_ * ny_);

    for (int k = 0; k < nt_; ++k)
      for (int i = 0; i < nx_; ++i)
        for (int j = 0; j < ny_; ++j) {
          FluidCellInfo cell;
          if (nf >= 1) cell.energy_density = ptr[idx(0, i, j, k)];
          if (nf >= 2) cell.temperature    = ptr[idx(1, i, j, k)];
          if (nf >= 3) cell.vx             = ptr[idx(2, i, j, k)];
          if (nf >= 4) cell.vy             = ptr[idx(3, i, j, k)];
          // Extended fields (if model produces them)
          if (nf >= 5) cell.entropy_density = ptr[idx(4, i, j, k)];
          if (nf >= 6) cell.pressure        = ptr[idx(5, i, j, k)];
          bulk_info.data.push_back(cell);
        }
  }

  // ── Medium query (FastHydro / any 3+1D Python hydro) ───────────────────────
  // FluidDynamics::GetHydroInfo() is a STUB: it range-checks and then never assigns
  // fluid_cell_info_ptr (the bulk_info.get() call is commented out, FluidDynamics.h:546-564).
  // Every caller dereferences the pointer unchecked -- LiquefierBase::filter_partons
  // (LiquefierBase.cc:126-128), Matter.cc:495,777-779, LBT.cc:702-706 -- so the stub is a
  // null dereference, not merely wrong data.  This override therefore ALWAYS allocates.
  //
  // Deliberately implemented in C++ and NOT exposed as a PYBIND11_OVERRIDE:
  //   * it is called once per parton per energy-loss timestep, so a Python dispatch would
  //     take the GIL and box a FluidCellInfo on every call;
  //   * a Python exception raised here would propagate through sigslot out of
  //     JetEnergyLoss::DoExecTime(), where nothing catches it.
  // The body is fully general because a Python hydro owns bulk_info completely.
  mutable std::size_t n_out_of_range_ = 0;

  // Snap a query coordinate that sits a hair outside the grid back onto the boundary.
  //
  // This is not cosmetic.  bulk_info's grid metadata is stored as Jetscape::real, which is
  // FLOAT (RealType.h), so a tau_min of 0.6 comes back as 0.6000000238418579.  A query at
  // exactly tau0 -- which is where Matter starts, <Eloss><tStart> = tau0 -- is then *below*
  // tau_min, CheckInRange returns 0, and EvolutionHistory::get() answers with a
  // default-constructed (i.e. vacuum) cell (FluidEvolutionHistory.cc, get()).  The whole
  // first frame, and the outermost cell on every axis, would be invisible: partons would
  // traverse vacuum for their first step and quench too little, with nothing in the log.
  //
  // The tolerance is a ten-thousandth of a cell: orders of magnitude above the float32
  // representation error at these magnitudes (~1e-7 fm) and far below anything physical, so
  // a parton that has genuinely left the fireball still sees vacuum and still gets counted.
  static double snap_(double v, double lo, double hi, double step) {
    const double tol = 1e-4 * std::fabs(step);
    if (v < lo && v > lo - tol) return lo;
    if (v > hi && v < hi + tol) return hi;
    return v;
  }

  void GetHydroInfo(Jetscape::real t, Jetscape::real x, Jetscape::real y, Jetscape::real z,
                    std::unique_ptr<FluidCellInfo> &fluid_cell_info_ptr) override {
    if (hydro_status != FINISHED ||
        (bulk_info.data.empty() && bulk_info.data_vector.empty())) {
      fluid_cell_info_ptr.reset(new FluidCellInfo());   // zero cell == vacuum
      ++n_out_of_range_;
      return;
    }
    try {
      double c0, c1, c2, c3;      // (tau, x, y, eta), or (t, x, y, z) if the store is t-z
      if (bulk_info.tau_eta_is_tz) {
        c0 = t; c1 = x; c2 = y; c3 = z;
      } else {
        const double tz = double(t) * t - double(z) * z;
        c0 = std::sqrt(std::max(0.0, tz));                       // tau
        c1 = x; c2 = y;
        c3 = (std::fabs(double(t)) > std::fabs(double(z)))       // eta
                 ? 0.5 * std::log((double(t) + z) / (double(t) - z))
                 : 0.0;
      }
      c0 = snap_(c0, bulk_info.tau_min, bulk_info.TauMax(), bulk_info.dtau);
      c1 = snap_(c1, bulk_info.x_min,   bulk_info.XMax(),   bulk_info.dx);
      c2 = snap_(c2, bulk_info.y_min,   bulk_info.YMax(),   bulk_info.dy);
      if (!bulk_info.boost_invariant)
        c3 = snap_(c3, bulk_info.eta_min, bulk_info.EtaMax(), bulk_info.deta);

      // CheckInRange never throws -- the throws are commented out -- so this is purely the
      // diagnostic counter. get() itself returns a zero cell when the point is outside.
      if (!bulk_info.CheckInRange(c0, c1, c2, c3))
        ++n_out_of_range_;

      fluid_cell_info_ptr.reset(new FluidCellInfo(bulk_info.get(c0, c1, c2, c3)));
    } catch (const std::exception &) {
      fluid_cell_info_ptr.reset(new FluidCellInfo());
    }
  }

  std::size_t get_out_of_range_count() const { return n_out_of_range_; }
  void reset_out_of_range_count() { n_out_of_range_ = 0; }

  // Convenience for tests: the same query, returned by value.
  FluidCellInfo get_hydro_cell_by_value(Jetscape::real t, Jetscape::real x,
                                        Jetscape::real y, Jetscape::real z) {
    std::unique_ptr<FluidCellInfo> p;
    GetHydroInfo(t, x, y, z, p);
    return *p;
  }

  // ── 3+1D fluid-cell storage ────────────────────────────────────────────────
  // store_fluid_cells_from_numpy() above is 2+1D only: it takes (F, nx, ny, ntau), has no
  // eta axis and never sets vz, so it is correct only for neta == 1.  These two take
  // (F, nx, ny, neta, ntau) and write in EvolutionHistory's own record order,
  //     rec = id_tau*nx*ny*neta + id_x*ny*neta + id_y*neta + id_eta
  // (FluidEvolutionHistory.h CellIndex).
  //
  // Both clear data_vector, data_info AND data first: clear_up_evolution_data() clears only
  // `data` (FluidEvolutionHistory.h:222), so a stale data_vector would otherwise survive.
  void check_3d_shape_(const py::buffer_info &buf, std::size_t n_names) const {
    if (buf.ndim != 5)
      throw std::runtime_error(
          "expected an array of shape (n_features, nx, ny, neta, ntau), got ndim=" +
          std::to_string(buf.ndim));
    if ((std::size_t)buf.shape[0] != n_names)
      throw std::runtime_error(
          "n_features (" + std::to_string(buf.shape[0]) + ") != len(names) (" +
          std::to_string(n_names) + ")");
    if (buf.shape[1] != bulk_info.nx || buf.shape[2] != bulk_info.ny ||
        buf.shape[3] != bulk_info.neta || buf.shape[4] != bulk_info.ntau)
      throw std::runtime_error(
          "array grid (" + std::to_string(buf.shape[1]) + "," + std::to_string(buf.shape[2]) +
          "," + std::to_string(buf.shape[3]) + "," + std::to_string(buf.shape[4]) +
          ") does not match bulk_info (" + std::to_string(bulk_info.nx) + "," +
          std::to_string(bulk_info.ny) + "," + std::to_string(bulk_info.neta) + "," +
          std::to_string(bulk_info.ntau) + "). Call set_hydro_grid_info() first.");
  }

  static void check_names_(const std::vector<std::string> &names) {
    for (const auto &n : names)
      if (ResolveEntryName(n) == ENTRY_INVALID)
        throw std::runtime_error(
            "unknown bulk_info field '" + n +
            "'. Legal names: energy_density, entropy_density, temperature, pressure, "
            "qgp_fraction, mu_b, mu_c, mu_s, vx, vy, vz, pi00..pi33, bulk_pi");
  }

  // SoA route (default): 4 B per stored field per cell.
  void store_fluid_cells_from_numpy_3d(
      py::array_t<float, py::array::c_style | py::array::forcecast> arr,
      std::vector<std::string> names) {
    auto buf = arr.request();
    check_names_(names);
    check_3d_shape_(buf, names.size());

    const int F = (int)names.size();
    const int nx_ = bulk_info.nx, ny_ = bulk_info.ny;
    const int neta_ = bulk_info.neta, nt_ = bulk_info.ntau;
    const std::size_t ncell = (std::size_t)nt_ * nx_ * ny_ * neta_;

    bulk_info.data.clear();
    bulk_info.data.shrink_to_fit();
    bulk_info.data_vector.assign(ncell * F, 0.0f);
    bulk_info.data_info = names;

    const float *ptr = static_cast<float *>(buf.ptr);
    // input is C-contiguous [f][ix][iy][ieta][itau]
    const std::size_t s_f = (std::size_t)nx_ * ny_ * neta_ * nt_;
    const std::size_t s_x = (std::size_t)ny_ * neta_ * nt_;
    const std::size_t s_y = (std::size_t)neta_ * nt_;
    const std::size_t s_e = (std::size_t)nt_;
    for (int k = 0; k < nt_; ++k)
      for (int i = 0; i < nx_; ++i)
        for (int j = 0; j < ny_; ++j)
          for (int l = 0; l < neta_; ++l) {
            const std::size_t rec =
                ((std::size_t)k * nx_ * ny_ * neta_) + (std::size_t)i * ny_ * neta_ +
                (std::size_t)j * neta_ + l;
            for (int f = 0; f < F; ++f)
              bulk_info.data_vector[rec * F + f] =
                  ptr[f * s_f + i * s_x + j * s_y + l * s_e + k];
          }
  }

  // AoS route: 112 B per cell regardless of how many fields are used.  Kept for small grids
  // and for consumers that want a populated FluidCellInfo without a data_info lookup.
  void store_fluid_cells_aos_3d(
      py::array_t<float, py::array::c_style | py::array::forcecast> arr,
      std::vector<std::string> names) {
    auto buf = arr.request();
    check_names_(names);
    check_3d_shape_(buf, names.size());

    const int F = (int)names.size();
    const int nx_ = bulk_info.nx, ny_ = bulk_info.ny;
    const int neta_ = bulk_info.neta, nt_ = bulk_info.ntau;

    bulk_info.data_vector.clear();
    bulk_info.data_vector.shrink_to_fit();
    bulk_info.data_info.clear();
    bulk_info.data.clear();
    bulk_info.data.reserve((std::size_t)nt_ * nx_ * ny_ * neta_);

    std::vector<EntryName> ids;
    ids.reserve(F);
    for (const auto &n : names) ids.push_back(ResolveEntryName(n));

    const float *ptr = static_cast<float *>(buf.ptr);
    const std::size_t s_f = (std::size_t)nx_ * ny_ * neta_ * nt_;
    const std::size_t s_x = (std::size_t)ny_ * neta_ * nt_;
    const std::size_t s_y = (std::size_t)neta_ * nt_;
    const std::size_t s_e = (std::size_t)nt_;
    for (int k = 0; k < nt_; ++k)
      for (int i = 0; i < nx_; ++i)
        for (int j = 0; j < ny_; ++j)
          for (int l = 0; l < neta_; ++l) {
            FluidCellInfo cell;
            for (int f = 0; f < F; ++f) {
              const float v = ptr[f * s_f + i * s_x + j * s_y + l * s_e + k];
              switch (ids[f]) {
                case ENTRY_ENERGY_DENSITY:  cell.energy_density  = v; break;
                case ENTRY_ENTROPY_DENSITY: cell.entropy_density = v; break;
                case ENTRY_TEMPERATURE:     cell.temperature     = v; break;
                case ENTRY_PRESSURE:        cell.pressure        = v; break;
                case ENTRY_QGP_FRACTION:    cell.qgp_fraction    = v; break;
                case ENTRY_MU_B:            cell.mu_B            = v; break;
                case ENTRY_MU_C:            cell.mu_C            = v; break;
                case ENTRY_MU_S:            cell.mu_S            = v; break;
                case ENTRY_VX:              cell.vx              = v; break;
                case ENTRY_VY:              cell.vy              = v; break;
                case ENTRY_VZ:              cell.vz              = v; break;
                case ENTRY_BULK_PI:         cell.bulk_Pi         = v; break;
                case ENTRY_PI00: cell.pi[0][0] = v; break;
                case ENTRY_PI01: cell.pi[0][1] = v; break;
                case ENTRY_PI02: cell.pi[0][2] = v; break;
                case ENTRY_PI03: cell.pi[0][3] = v; break;
                case ENTRY_PI11: cell.pi[1][1] = v; break;
                case ENTRY_PI12: cell.pi[1][2] = v; break;
                case ENTRY_PI13: cell.pi[1][3] = v; break;
                case ENTRY_PI22: cell.pi[2][2] = v; break;
                case ENTRY_PI23: cell.pi[2][3] = v; break;
                case ENTRY_PI33: cell.pi[3][3] = v; break;
                default: break;
              }
            }
            bulk_info.data.push_back(cell);
          }
  }

  // ── Status helpers ─────────────────────────────────────────────────────────
  void set_hydro_status_finished()    { hydro_status = FINISHED; }
  void set_hydro_status_initialized() { hydro_status = INITIALIZED; }
  int  get_hydro_status_int() const   { return static_cast<int>(hydro_status); }

  // ── Freeze-out surface convenience wrapper ─────────────────────────────────
  // Calls the public FluidDynamics::FindAConstantTemperatureSurface() and
  // stores the result in the protected surfaceCellVector_ member so that
  // the framework can retrieve it via getSurfaceCellVector().
  void find_freezeout_surface(Jetscape::real T_sw, Jetscape::real dtau = 0.,
                              Jetscape::real dx = 0., Jetscape::real deta = 0.) {
    clearSurfaceCellVector();
    SurfaceFinderParams params;
    params.T_sw = T_sw;
    params.dtau = dtau;
    params.dx = dx;
    params.deta = deta;
    FindSurfaceFromEvolution(params, surfaceCellVector_);
  }
};

// ─────────────────────────────────────────────────────────────────────────────

void bind_fluid_dynamics(py::module_ &m) {

  // ── Parameter ─────────────────────────────────────────────────────────────
  // Minimal binding — Python InitializeHydro overrides receive this object
  // but typically ignore it (reading config from Python dicts instead).
  py::class_<Parameter>(m, "Parameter")
      .def(py::init<>())
      .def_property("hydro_input_filename",
          [](const Parameter &p) -> std::string {
            return p.hydro_input_filename ? std::string(p.hydro_input_filename)
                                          : "";
          },
          [](Parameter &p, const std::string & /*unused*/) {
            // Setting a char* from Python is intentionally not supported;
            // Python hydro modules should read config from Python dicts.
          });

  // ── HydroStatus enum ──────────────────────────────────────────────────────
  py::enum_<HydroStatus>(m, "HydroStatus")
      .value("NOT_START",  HydroStatus::NOT_START)
      .value("INITIALIZED",HydroStatus::INITIALIZED)
      .value("EVOLVING",   HydroStatus::EVOLVING)
      .value("FINISHED",   HydroStatus::FINISHED)
      .value("ERROR",      HydroStatus::ERROR)
      .export_values();

  // ── PreequilibriumDynamics ─────────────────────────────────────────────────
  // Concrete implementations (e.g. FreestreamMilne) are created via
  // create_module().  Python code accesses the stress-energy fields as numpy
  // arrays via the get_*_numpy() helpers below.
  py::class_<PreequilibriumDynamics, JetScapeModuleBase,
             std::shared_ptr<PreequilibriumDynamics>>(
      m, "PreequilibriumDynamics",
      R"pbdoc(
        Interface for pre-equilibrium dynamics modules.
        All stress-energy-tensor fields (e_, P_, ux_, uy_, ueta_, pi??_, bulk_Pi_)
        are public std::vector<double> members — directly accessible as zero-copy
        numpy arrays via the get_*_numpy() methods.
      )pbdoc")
      .def("GetPreequilibriumStartTime",
           &PreequilibriumDynamics::GetPreequilibriumStartTime,
           "Proper time tau_0 at the start of pre-equilibrium [fm/c].")
      .def("GetPreequilibriumEndTime",
           &PreequilibriumDynamics::GetPreequilibriumEndTime,
           "Proper time tau at the end of pre-equilibrium [fm/c].")
      .def("get_ntau",   &PreequilibriumDynamics::get_ntau)
      .def("get_number_of_fluid_cells",
           &PreequilibriumDynamics::get_number_of_fluid_cells)
      // ── Zero-copy numpy views for all public stress-energy fields ──────────
      // All fields are 1D arrays of length (nx * ny) in the pre-equilibrium grid.
#define PREEQ_FIELD(name, member, doc)                                       \
  .def(                                                                       \
      "get_" #name "_numpy",                                                  \
      [](PreequilibriumDynamics &p) -> py::array_t<double> {                 \
        if (p.member.empty())                                                 \
          return py::array_t<double>({0}, {sizeof(double)}, nullptr);        \
        return py::array_t<double>(                                           \
            {(py::ssize_t)p.member.size()},                                  \
            {sizeof(double)},                                                 \
            p.member.data(),                                                  \
            py::cast(&p));                                                    \
      },                                                                      \
      py::return_value_policy::reference_internal, doc)
      PREEQ_FIELD(e,       e_,       "Energy density [GeV/fm^3], shape (nx*ny,).")
      PREEQ_FIELD(P,       P_,       "Pressure [GeV/fm^3], shape (nx*ny,).")
      PREEQ_FIELD(utau,    utau_,    "u^tau, shape (nx*ny,).")
      PREEQ_FIELD(ux,      ux_,      "u^x, shape (nx*ny,).")
      PREEQ_FIELD(uy,      uy_,      "u^y, shape (nx*ny,).")
      PREEQ_FIELD(ueta,    ueta_,    "u^eta, shape (nx*ny,).")
      PREEQ_FIELD(pi00,    pi00_,    "pi^{00} [GeV/fm^3], shape (nx*ny,).")
      PREEQ_FIELD(pi01,    pi01_,    "pi^{01}, shape (nx*ny,).")
      PREEQ_FIELD(pi02,    pi02_,    "pi^{02}, shape (nx*ny,).")
      PREEQ_FIELD(pi03,    pi03_,    "pi^{03}, shape (nx*ny,).")
      PREEQ_FIELD(pi11,    pi11_,    "pi^{11}, shape (nx*ny,).")
      PREEQ_FIELD(pi12,    pi12_,    "pi^{12}, shape (nx*ny,).")
      PREEQ_FIELD(pi13,    pi13_,    "pi^{13}, shape (nx*ny,).")
      PREEQ_FIELD(pi22,    pi22_,    "pi^{22}, shape (nx*ny,).")
      PREEQ_FIELD(pi23,    pi23_,    "pi^{23}, shape (nx*ny,).")
      PREEQ_FIELD(pi33,    pi33_,    "pi^{33}, shape (nx*ny,).")
      PREEQ_FIELD(bulk_Pi, bulk_Pi_, "Bulk pressure Pi [GeV/fm^3], shape (nx*ny,).")
#undef PREEQ_FIELD
      ;

  // ── FluidDynamics ──────────────────────────────────────────────────────────
  // Registered with the PyFluidDynamics trampoline so that Python subclasses
  // (e.g. PyFNOHydro) can override InitializeHydro() and EvolveHydro().
  py::class_<FluidDynamics, JetScapeModuleBase, PyFluidDynamics,
             std::shared_ptr<FluidDynamics>>(m, "FluidDynamics",
      R"pbdoc(
        Base class for hydrodynamics modules.

        Subclass this in Python to inject a custom hydro implementation (e.g.
        an FNO model) into the JETSCAPE pipeline:

            class MyHydro(pyjetscape.FluidDynamics):
                def InitializeHydro(self, params):
                    ...  # read config
                def EvolveHydro(self):
                    ...  # run model, call store_fluid_cells_from_numpy()

        The trampoline class PyFluidDynamics (transparent to Python) dispatches
        virtual C++ calls to the Python override automatically.
      )pbdoc")
      .def(py::init<>())
      // ── JETSCAPE virtual interface ─────────────────────────────────────────
      .def("Init", &FluidDynamics::Init,
           "Read XML, retrieve ini/pre_eq pointers, call InitializeHydro().")
      .def("Exec", &FluidDynamics::Exec,
           "Call EvolveHydro().")
      .def("Clear", &FluidDynamics::Clear)
      // ── Bulk-info preservation across ClearTasks() ─────────────────────────
      // JetScapeTask::ClearTasks() calls Clear() at the end of every event,
      // which calls clear_up_evolution_data() and wipes bulk_info.data.
      // Set preserve_bulk_info(True) to skip that so Python can inspect the
      // data after js.Exec() returns.  EvolveHydro() must then call
      // clear_up_evolution_data() itself at the start for multi-event safety.
      .def("set_preserve_bulk_info",
           [](PyFluidDynamics &fd, bool preserve) {
             fd.preserve_bulk_info_ = preserve;
           },
           "When True, Clear() will not wipe bulk_info.data. "
           "Lets Python read bulk_info after js.Exec() returns.",
           py::arg("preserve"))
      .def("get_preserve_bulk_info",
           [](PyFluidDynamics &fd) { return fd.preserve_bulk_info_; })
      .def("InitializeHydro", &FluidDynamics::InitializeHydro,
           "Override in Python to initialise hydro (called from Init()).",
           py::arg("parameter_list"))
      .def("EvolveHydro", &FluidDynamics::EvolveHydro,
           "Override in Python to run hydro evolution (called from Exec()).")
      // ── Status ─────────────────────────────────────────────────────────────
      .def("GetHydroStatus",  &FluidDynamics::GetHydroStatus)
      .def("GetHydroStartTime",
           [](FluidDynamics &f) {
             double tau0 = 0.;
             f.GetHydroStartTime(tau0);
             return tau0;
           })
      .def("GetHydroEndTime", &FluidDynamics::GetHydroEndTime)
      .def("GetHydroFreezeOutTemperature",
           &FluidDynamics::GetHydroFreezeOutTemperature)
      // ── Bulk info (read-only public accessor) ──────────────────────────────
      .def("get_bulk_info",
           &FluidDynamics::get_bulk_info,
           py::return_value_policy::reference_internal,
           "Return a reference to the EvolutionHistory (read-only).")
      // ── Surface cells ──────────────────────────────────────────────────────
      .def("clearSurfaceCellVector", &FluidDynamics::clearSurfaceCellVector,
           "Clear the freeze-out surface cell list.")
      .def("getSurfaceCellVector",
           [](FluidDynamics &f) {
             std::vector<SurfaceCellInfo> v;
             f.getSurfaceCellVector(v);
             return v;
           },
           "Return a copy of the freeze-out surface cell vector.")
      .def("FindAConstantTemperatureSurface",
           [](FluidDynamics &f, Jetscape::real T_sw) {
             std::vector<SurfaceCellInfo> v;
             f.FindAConstantTemperatureSurface(T_sw, v);
             return v;
           },
           "Find the constant-temperature surface and return list of cells.",
           py::arg("T_sw"))
      // ── PyFluidDynamics-specific helpers (lambda wrappers required because
      //    these methods live on the trampoline, not on FluidDynamics itself) ──
      .def("get_ini_pointer",
           [](PyFluidDynamics &fd) { return fd.get_ini_pointer(); },
           py::return_value_policy::reference_internal,
           "Return shared_ptr to the InitialState module.")
      .def("get_preeq_pointer",
           [](PyFluidDynamics &fd) { return fd.get_preeq_pointer(); },
           py::return_value_policy::reference_internal,
           "Return shared_ptr to the PreequilibriumDynamics module.")
      .def("get_bulk_info_mutable",
           [](PyFluidDynamics &fd) -> EvolutionHistory & {
             return fd.get_bulk_info_mutable();
           },
           py::return_value_policy::reference_internal,
           "Return a mutable reference to the EvolutionHistory.")
      .def("get_surface_cells",
           [](PyFluidDynamics &fd) -> std::vector<SurfaceCellInfo> & {
             return fd.get_surface_cells();
           },
           py::return_value_policy::reference_internal,
           "Return a mutable reference to the surface cell vector.")
      // ── Hydro grid setup ───────────────────────────────────────────────────
      .def("set_hydro_grid_info",
           [](PyFluidDynamics &fd,
              float tau_min, float dtau, int ntau,
              float x_min,   float dx,   int nx,
              float y_min,   float dy,   int ny,
              float eta_min, float deta, int neta,
              bool boost_inv, bool tau_eta_is_tz) {
             fd.set_hydro_grid_info(tau_min, dtau, ntau,
                                    x_min, dx, nx,
                                    y_min, dy, ny,
                                    eta_min, deta, neta,
                                    boost_inv, tau_eta_is_tz);
           },
           R"pbdoc(
             Set bulk_info grid metadata.  Must be called before
             store_fluid_cells_from_numpy() in EvolveHydro().

             Parameters
             ----------
             tau_min, dtau, ntau : float, float, int
             x_min, dx, nx       : float, float, int
             y_min, dy, ny       : float, float, int
             eta_min, deta, neta : float, float, int
             boost_inv           : bool, default True
             tau_eta_is_tz       : bool, default False
           )pbdoc",
           py::arg("tau_min"), py::arg("dtau"), py::arg("ntau"),
           py::arg("x_min"),   py::arg("dx"),   py::arg("nx"),
           py::arg("y_min"),   py::arg("dy"),   py::arg("ny"),
           py::arg("eta_min"), py::arg("deta"), py::arg("neta"),
           py::arg("boost_inv") = true,
           py::arg("tau_eta_is_tz") = false)
      // ── Fluid cell storage ─────────────────────────────────────────────────
      .def("store_fluid_cells_from_numpy",
           [](PyFluidDynamics &fd,
              py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
             fd.store_fluid_cells_from_numpy(arr);
           },
           R"pbdoc(
             Populate bulk_info.data from a numpy array.

             Parameters
             ----------
             arr : np.ndarray, shape (n_features, nx, ny, ntau), dtype float32
                 Feature layout (same as FnoHydro C++ convention):
                   n_features == 4: [energy_density, temperature, vx, vy]
                   n_features == 3: [energy_density, vx, vy]
                 Energy density must already be in physical units [GeV/fm^3]
                 (apply inverse tau-normalisation in Python before calling).

             Notes
             -----
             * Call set_hydro_grid_info() before this method.
             * The array dimensions (nx, ny, ntau) must match the grid info.
           )pbdoc",
           py::arg("arr"))
      // ── 3+1D storage (FastHydro); the 2+1D version above is neta == 1 only ──
      .def("store_fluid_cells_from_numpy_3d",
           [](PyFluidDynamics &fd,
              py::array_t<float, py::array::c_style | py::array::forcecast> arr,
              std::vector<std::string> names) {
             fd.store_fluid_cells_from_numpy_3d(std::move(arr), std::move(names));
           },
           R"pbdoc(
             Fill bulk_info.data_vector from a (n_features, nx, ny, neta, ntau) float32 array.

             `names` must be as long as n_features and each entry must be one that
             ResolveEntryName() accepts: energy_density, entropy_density, temperature,
             pressure, qgp_fraction, mu_b, mu_c, mu_s, vx, vy, vz, pi00..pi33, bulk_pi.

             Cells are written in EvolutionHistory record order
             (tau, x, y, eta).  data_vector, data_info and data are all cleared first --
             clear_up_evolution_data() clears only `data`.

             Call set_hydro_grid_info() first; the grid is checked against it.

             This is the memory-efficient route: 4 bytes per stored field per cell, against
             112 bytes per cell for store_fluid_cells_aos_3d().
           )pbdoc",
           py::arg("arr"), py::arg("names"))
      .def("store_fluid_cells_aos_3d",
           [](PyFluidDynamics &fd,
              py::array_t<float, py::array::c_style | py::array::forcecast> arr,
              std::vector<std::string> names) {
             fd.store_fluid_cells_aos_3d(std::move(arr), std::move(names));
           },
           "Same input as store_fluid_cells_from_numpy_3d(), but fills bulk_info.data "
           "(array of FluidCellInfo) instead of data_vector.",
           py::arg("arr"), py::arg("names"))
      // ── Medium query ───────────────────────────────────────────────────────
      .def("get_hydro_cell",
           [](PyFluidDynamics &fd, Jetscape::real t, Jetscape::real x,
              Jetscape::real y, Jetscape::real z) {
             return fd.get_hydro_cell_by_value(t, x, y, z);
           },
           "GetHydroInfo() at Cartesian (t, x, y, z), returned by value. "
           "Never raises and never returns null: out-of-range gives a zero cell.",
           py::arg("t"), py::arg("x"), py::arg("y"), py::arg("z"))
      .def("get_out_of_range_count",
           [](PyFluidDynamics &fd) { return fd.get_out_of_range_count(); },
           "Number of GetHydroInfo() queries that fell outside the stored grid (and so saw "
           "vacuum). CheckInRange never throws, so this is the only signal.")
      .def("reset_out_of_range_count",
           [](PyFluidDynamics &fd) { fd.reset_out_of_range_count(); })
      // ── Liquefier ──────────────────────────────────────────────────────────
      .def("add_a_liquefier", &FluidDynamics::add_a_liquefier,
           "Attach a LiquefierBase (or CausalLiquefier) to this hydro.",
           py::arg("liquefier"))
      .def("get_liquefier",
           [](FluidDynamics &fd) -> std::shared_ptr<LiquefierBase> {
             return fd.get_liquefier().lock();     // None if never attached
           },
           "The attached liquefier, or None.")
      .def("SetHydroStartTime", &FluidDynamics::SetHydroStartTime,
           "Set hydro_tau_0 (uninitialised by the FluidDynamics constructor).",
           py::arg("tau0"))
      // ── Status control ─────────────────────────────────────────────────────
      .def("set_hydro_status_finished",
           [](PyFluidDynamics &fd) { fd.set_hydro_status_finished(); },
           "Mark hydro as FINISHED so that GetHydroInfo() works.")
      .def("set_hydro_status_initialized",
           [](PyFluidDynamics &fd) { fd.set_hydro_status_initialized(); },
           "Mark hydro as INITIALIZED.")
      .def("get_hydro_status_int",
           [](PyFluidDynamics &fd) { return fd.get_hydro_status_int(); },
           "Return the raw HydroStatus integer.")
      // ── Freeze-out surface ─────────────────────────────────────────────────
      .def("find_freezeout_surface",
           [](PyFluidDynamics &fd, Jetscape::real T_sw, Jetscape::real dtau,
              Jetscape::real dx, Jetscape::real deta) {
             fd.find_freezeout_surface(T_sw, dtau, dx, deta);
           },
           R"pbdoc(
             Find the iso-temperature freeze-out surface and store it.

             Calls clearSurfaceCellVector() + FindAConstantTemperatureSurface()
             and stores the result in surfaceCellVector_ so the framework can
             retrieve it via getSurfaceCellVector().  Requires hydro_status
             to be FINISHED and bulk_info.data to be populated.

             Parameters
             ----------
             T_sw : float
                 Switch (freeze-out) temperature [GeV].
             dtau, dx, deta : float, optional
                 Cornelius lattice spacing (dy = dx).  0 keeps SurfaceFinder's
                 defaults (0.1 fm, 0.2 fm, 0.2).
           )pbdoc",
           py::arg("T_sw"), py::arg("dtau") = 0., py::arg("dx") = 0.,
           py::arg("deta") = 0.)
      // ── clear evolution data ───────────────────────────────────────────────
      .def("clear_up_evolution_data",
           &FluidDynamics::clear_up_evolution_data,
           "Clear the FluidCellInfo data in bulk_info.")
      // ── SetId convenience (needed for PyFNOHydro) ──────────────────────────
      .def("SetId", &FluidDynamics::SetId,
           "Set the module identifier string (e.g. 'PyFNOHydro').",
           py::arg("id"));
}
