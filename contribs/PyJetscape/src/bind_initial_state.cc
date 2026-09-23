/*******************************************************************************
 * bind_initial_state.cc
 *
 * Binds InitialState:
 *   - Grid accessors (GetXMax/Step/Size, GetYMax/Step/Size, GetZMax/Step/Size)
 *   - GetEntropyDensityDistribution() → flat std::vector<double>
 *   - get_entropy_density_numpy() → 2D numpy array (nx, ny) for boost-invariant
 *   - GetNumOfBinaryCollisions() → flat std::vector<double>
 *   - Event-level quantities (GetNpart, GetNcoll, GetTotalEntropy, …)
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "InitialState.h"
#include "JetScapeModuleBase.h"

namespace py = pybind11;
using namespace Jetscape;

// ── PyInitialState trampoline ────────────────────────────────────────────────
// Lets Python subclasses override Init(), Exec(), and Clear() so that a custom
// initial condition (e.g. a Gaussian profile) can be injected into the pipeline
// without an external binary such as trento.
//
// set_entropy_density_from_numpy() populates the protected member
// entropy_density_distribution_ directly from a numpy array, following the
// storage convention used by TrentoInitial:
//   boost-invariant (nz=1):  for y { for x }   ← flat shape (ny*nx,) or (ny,nx)
class PyInitialState : public InitialState {
public:
  using InitialState::InitialState;

  void Init() override {
    PYBIND11_OVERRIDE(void, InitialState, Init);
  }
  void Exec() override {
    PYBIND11_OVERRIDE(void, InitialState, Exec);
  }
  void Clear() override {
    PYBIND11_OVERRIDE(void, InitialState, Clear);
  }

  // Populate entropy_density_distribution_ from a flat or 2-D numpy array.
  // For the boost-invariant case (nz = 1) the expected shape is (ny, nx) or
  // (ny*nx,); any C-contiguous layout is accepted and the data is copied flat.
  void set_entropy_density_from_numpy(
      py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    size_t n = 1;
    for (auto &s : buf.shape)
      n *= static_cast<size_t>(s);
    const double *ptr = static_cast<const double *>(buf.ptr);
    entropy_density_distribution_.assign(ptr, ptr + n);
  }

  // Populate num_of_binary_collisions_, the Ta*Tb density that
  // InitialState::SampleABinaryCollisionPoint() draws hard-scattering vertices from.
  // Without it every jet starts at (0,0,0) and the framework only warns.
  void set_num_of_binary_collisions_from_numpy(
      py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    size_t n = 1;
    for (auto &sh : buf.shape)
      n *= static_cast<size_t>(sh);
    const double *ptr = static_cast<const double *>(buf.ptr);
    num_of_binary_collisions_.assign(ptr, ptr + n);
  }
};

// ─────────────────────────────────────────────────────────────────────────────

void bind_initial_state(py::module_ &m) {

  // InitialState is a C++ abstract-ish base — concrete implementations are
  // registered via RegisterJetScapeModule<T> and created through create_module().
  // The PyInitialState trampoline allows Python subclasses to override Init(),
  // Exec(), and Clear() and to fill the IC grid via set_entropy_density_from_numpy().
  py::class_<InitialState, JetScapeModuleBase, PyInitialState,
             std::shared_ptr<InitialState>>(m, "InitialState",
      R"pbdoc(
        Interface for initial-state physics modules (e.g. TrentoInitial).

        Provides the entropy/binary-collision density grids that seed
        pre-equilibrium and hydro evolution.

        Python subclasses can override Init(), Exec(), and Clear() and
        populate the IC grid with set_entropy_density_from_numpy() instead
        of running an external binary.
      )pbdoc")
      .def(py::init<>())
      // ── Grid range accessors ───────────────────────────────────────────────
      .def("GetXMax",  &InitialState::GetXMax,
           "Maximum |x| of the nuclear profile grid [fm].")
      .def("GetXStep", &InitialState::GetXStep,
           "Step size dx [fm].")
      .def("GetYMax",  &InitialState::GetYMax,
           "Maximum |y| of the nuclear profile grid [fm].")
      .def("GetYStep", &InitialState::GetYStep,
           "Step size dy [fm].")
      .def("GetZMax",  &InitialState::GetZMax,
           "Maximum |z or eta| of the nuclear profile grid.")
      .def("GetZStep", &InitialState::GetZStep,
           "Step size dz (or d-eta) in z direction.")
      .def("GetXSize", &InitialState::GetXSize,
           "Number of grid points in x: ceil(2*xmax/dx).")
      .def("GetYSize", &InitialState::GetYSize,
           "Number of grid points in y: ceil(2*ymax/dy).")
      .def("GetZSize", &InitialState::GetZSize,
           "Number of grid points in z (1 for boost-invariant).")
      // ── Density distributions ──────────────────────────────────────────────
      .def("GetEntropyDensityDistribution",
           &InitialState::GetEntropyDensityDistribution,
           R"pbdoc(
             Return the entropy density distribution as a flat list.
             Storage order: for z { for y { for x } }.
             Length = GetXSize() * GetYSize() * GetZSize().
           )pbdoc")
      .def("GetNumOfBinaryCollisions",
           &InitialState::GetNumOfBinaryCollisions,
           R"pbdoc(
             Return the un-normalised T_A*T_B binary-collision density as a flat list.
             Same storage order as GetEntropyDensityDistribution().
           )pbdoc")
      // ── Convenience: 2D numpy view (boost-invariant case) ─────────────────
      .def("get_entropy_density_numpy",
           [](InitialState &ini) -> py::array_t<double> {
             const auto &dist = ini.GetEntropyDensityDistribution();
             int nx = ini.GetXSize();
             int ny = ini.GetYSize();
             if ((int)dist.size() < nx * ny)
               throw std::runtime_error(
                   "Entropy density distribution has fewer entries than nx*ny. "
                   "Make sure Init()+Exec() have been called on the module.");
             // Storage order is for y { for x }, i.e. row = y-index.
             // Return shape (ny, nx) so arr[iy, ix] == dist[iy*nx + ix].
             py::array_t<double> arr({ny, nx});
             auto r = arr.mutable_unchecked<2>();
             for (int iy = 0; iy < ny; ++iy)
               for (int ix = 0; ix < nx; ++ix)
                 r(iy, ix) = dist[iy * nx + ix];
             return arr;
           },
           R"pbdoc(
             Return the entropy density as a 2D numpy array of shape (ny, nx).
             Only valid for the boost-invariant (2D) case (GetZSize() == 1).
             Equivalent to:
               dist = GetEntropyDensityDistribution()
               arr  = np.array(dist).reshape(ny, nx)
           )pbdoc")
      .def("set_num_of_binary_collisions_from_numpy",
           [](PyInitialState &ini,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             ini.set_num_of_binary_collisions_from_numpy(std::move(arr));
           },
           R"pbdoc(
             Set the Ta*Tb binary-collision density that hard processes sample vertices from.

             Same C-order (nx, ny, neta) ravel as set_entropy_density_from_numpy().  Without
             it, InitialState::SampleABinaryCollisionPoint() warns and puts every jet at the
             origin, so all showers start at the fireball centre.

             NOTE the half-cell offset: CoordFromIdx() maps index ix to
             ``-grid_max_x + ix*grid_step_x`` (InitialState.cc:75-77), i.e. MUSIC's
             ``-n*d/2 + i*d`` axis, while the energy-density grid this contribution uses is
             cell-centred at ``-(n-1)*d/2 + i*d``.  Sampled vertices therefore sit half a cell
             from the matching energy-density node.
           )pbdoc",
           py::arg("arr"))
      .def("sample_binary_collision_point",
           [](InitialState &ini) {
             double t = 0.0, x = 0.0, y = 0.0, z = 0.0;
             ini.SampleABinaryCollisionPoint(t, x, y, z);
             return py::make_tuple(t, x, y, z);
           },
           R"pbdoc(
             Draw one hard-scattering vertex, exactly as the hard process does.

             Returns (t, x, y, z) in fm.  With no density set this returns the origin (and the
             framework prints a warning), which is the whole reason
             set_num_of_binary_collisions_from_numpy() exists.

             Coordinates come from CoordFromIdx, i.e. MUSIC's ``-grid_max + i*step`` axis.
           )pbdoc")
      // ── 3+1D numpy view ────────────────────────────────────────────────────
      .def("get_entropy_density_numpy_3d",
           [](InitialState &ini) -> py::array_t<double> {
             const auto &dist = ini.GetEntropyDensityDistribution();
             const int nx = ini.GetXSize(), ny = ini.GetYSize(), nz = ini.GetZSize();
             const std::size_t want = (std::size_t)nx * ny * nz;
             if (dist.size() < want)
               throw std::runtime_error(
                   "entropy density distribution has " + std::to_string(dist.size()) +
                   " entries, need nx*ny*neta = " + std::to_string(want) +
                   ". Has Exec() run, and do SetRanges/SetSteps match the array?");
             py::array_t<double> arr({nx, ny, nz});
             auto r = arr.mutable_unchecked<3>();
             for (int ix = 0; ix < nx; ++ix)
               for (int iy = 0; iy < ny; ++iy)
                 for (int ie = 0; ie < nz; ++ie)
                   r(ix, iy, ie) = dist[((std::size_t)ix * ny + iy) * nz + ie];
             return arr;
           },
           R"pbdoc(
             The initial-state grid as a 3D numpy array of shape (nx, ny, neta).

             Storage order is the C-order (nx, ny, neta) ravel,
                 idx = (ny*neta)*ix + neta*iy + ieta
             which is what set_entropy_density_from_numpy() copies in verbatim and what
             MUSIC's Initial_profile 42 reads.

             NOTE: get_entropy_density_numpy() (the 2D convenience view) instead assumes
             dist[iy*nx + ix] and is documented as valid only for GetZSize() == 1.  For a
             square boost-invariant grid the two differ by a transpose, so do not mix them.
           )pbdoc")
      // ── Event-level scalars ────────────────────────────────────────────────
      .def("GetNpart",           &InitialState::GetNpart,
           "Number of participant nucleons (-1 if not available).")
      .def("GetNcoll",           &InitialState::GetNcoll,
           "Number of binary collisions (-1 if not available).")
      .def("GetTotalEntropy",    &InitialState::GetTotalEntropy,
           "Total entropy of the event (-1 if not available).")
      .def("GetEventCentrality", &InitialState::GetEventCentrality,
           "Event centrality class (-1 if not available).")
      .def("GetEventId",         &InitialState::GetEventId,
           "Current event ID.")
      .def("SetEventId",         &InitialState::SetEventId,
           py::arg("event_id"))
      // ── Geometry setters (useful for testing without XML) ─────────────────
      .def("SetRanges", &InitialState::SetRanges,
           "Set grid extents (xmax, ymax, zmax).",
           py::arg("xmax"), py::arg("ymax"), py::arg("zmax"))
      .def("SetSteps", &InitialState::SetSteps,
           "Set grid step sizes (dx, dy, dz).",
           py::arg("dx"), py::arg("dy"), py::arg("dz"))
      // ── Python IC injection ───────────────────────────────────────────────
      .def("set_entropy_density_from_numpy",
           [](PyInitialState &ini,
              py::array_t<double,
                          py::array::c_style | py::array::forcecast> arr) {
             ini.set_entropy_density_from_numpy(arr);
           },
           R"pbdoc(
             Populate entropy_density_distribution_ from a numpy array.

             Call this from a Python InitialState subclass inside Exec() to
             inject a custom initial condition (e.g. a Gaussian profile)
             without running an external program such as trento.

             Parameters
             ----------
             arr : np.ndarray, shape (ny, nx) or (ny*nx,), dtype float64
                 Entropy (or energy) density values in C-contiguous order:
                 boost-invariant storage is ``for y { for x }``, so
                 ``arr[iy, ix]`` maps to index ``iy * nx + ix``.
           )pbdoc",
           py::arg("arr"));
}
