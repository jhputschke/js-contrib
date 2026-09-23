/*******************************************************************************
 * bind_liquefier.cc
 *
 * Binds the jet-source ("liquefier") layer so a Python hydro can consume the droplets
 * that Matter/LBT deposit:
 *
 *   Droplet         value object: Milne position (tau,x,y,eta) + Cartesian momentum (E,px,py,pz)
 *   LiquefierBase   the droplet list, plus droplets_numpy()/add_droplets_numpy()
 *   CausalLiquefier the causal-diffusion kernel, both constructors
 *
 * Droplets are produced centrally by the framework -- JetEnergyLoss::DoExecTime() calls
 * liquefier->add_hydro_sources(pIn, pOut) (JetEnergyLoss.cc:374-377) -- not by Matter or LBT
 * themselves.  So all a Python hydro has to do is read the list out after energy loss.
 *
 * The (M,8) column order of droplets_numpy() is
 *     tau, x, y, eta, E, px, py, pz
 * which is Droplet::xmu ++ Droplet::pmu as filled by LiquefierBase::add_hydro_sources
 * (LiquefierBase.cc:200-225), and is byte-for-byte the column order of
 * fast_data.liquefier.droplets.COLUMNS.  No conversion is needed on either side.
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <memory>

#include "LiquefierBase.h"
#include "CausalLiquefier.h"

namespace py = pybind11;
using namespace Jetscape;

//: droplets_numpy() / add_droplets_numpy() column order
static const char *kDropletColumns[8] = {"tau", "x", "y", "eta", "E", "px", "py", "pz"};

void bind_liquefier(py::module_ &m) {

  // ── Droplet ───────────────────────────────────────────────────────────────
  py::class_<Droplet>(m, "Droplet",
                      "A unit of four-momentum handed from the jet to the medium.\n"
                      "Position is Milne (tau, x, y, eta); momentum is Cartesian (E, px, py, pz).")
      .def(py::init<>())
      .def(py::init<std::array<Jetscape::real, 4>, std::array<Jetscape::real, 4>>(),
           py::arg("xmu"), py::arg("pmu"))
      .def("get_xmu", &Droplet::get_xmu, "(tau, x, y, eta)")
      .def("get_pmu", &Droplet::get_pmu, "(E, px, py, pz)")
      .def("__repr__", [](const Droplet &d) {
        auto x = d.get_xmu();
        auto p = d.get_pmu();
        return "<Droplet x=(" + std::to_string(x[0]) + "," + std::to_string(x[1]) + "," +
               std::to_string(x[2]) + "," + std::to_string(x[3]) + ") E=" +
               std::to_string(p[0]) + ">";
      });

  // ── LiquefierBase ─────────────────────────────────────────────────────────
  py::class_<LiquefierBase, std::shared_ptr<LiquefierBase>>(m, "LiquefierBase")
      .def("add_a_droplet", &LiquefierBase::add_a_droplet, py::arg("droplet"))
      .def("get_a_droplet", &LiquefierBase::get_a_droplet, py::arg("idx"))
      .def("get_dropletlist_size", &LiquefierBase::get_dropletlist_size)
      .def("get_dropletlist_total_energy", &LiquefierBase::get_dropletlist_total_energy)
      .def("get_drop_stat", &LiquefierBase::get_drop_stat,
           "pstat of a parton absorbed into the fluid (-11).")
      .def("get_miss_stat", &LiquefierBase::get_miss_stat,
           "pstat of the energy-momentum-conservation filler parton (-13).")
      .def("get_neg_stat", &LiquefierBase::get_neg_stat,
           "pstat of a negative / back-reaction parton (-17).")
      .def("ClearTask", &LiquefierBase::ClearTask,
           "Empty the droplet list. FluidDynamics::ClearTask() is normally what calls this; "
           "a Python hydro that overrides Clear() must call it itself, or droplets "
           "accumulate across events.")
      .def("get_GetHydroCellSignalConnected",
           &LiquefierBase::get_GetHydroCellSignalConnected,
           "True once the framework has wired this liquefier to a hydro. "
           "filter_partons() needs that signal to boost partons into the fluid rest frame.")
      .def_property_readonly_static("COLUMNS", [](py::object) {
        py::tuple t(8);
        for (int i = 0; i < 8; ++i) t[i] = kDropletColumns[i];
        return t;
      }, "Column order of droplets_numpy(): (tau, x, y, eta, E, px, py, pz).")
      // ── numpy bridge ────────────────────────────────────────────────────────
      .def("droplets_numpy",
           [](const LiquefierBase &liq) {
             const int n = liq.get_dropletlist_size();
             py::array_t<double> out({(py::ssize_t)n, (py::ssize_t)8});
             auto r = out.mutable_unchecked<2>();
             for (int i = 0; i < n; ++i) {
               const Droplet d = liq.get_a_droplet(i);
               const auto x = d.get_xmu();
               const auto p = d.get_pmu();
               for (int k = 0; k < 4; ++k) {
                 r(i, k) = static_cast<double>(x[k]);
                 r(i, 4 + k) = static_cast<double>(p[k]);
               }
             }
             return out;
           },
           R"pbdoc(
             The droplet list as an (M, 8) float64 array.

             Columns are (tau, x, y, eta, E, px, py, pz) -- exactly
             fast_data.liquefier.droplets.COLUMNS, so the result is a DropletArray payload
             with no conversion.

             float64 is emitted even though Jetscape::real is float: the values have already
             been rounded to float, and keeping them wide avoids a second rounding.
           )pbdoc")
      .def("add_droplets_numpy",
           [](LiquefierBase &liq,
              py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
             auto buf = arr.request();
             if (buf.ndim != 2 || buf.shape[1] != 8)
               throw std::runtime_error(
                   "add_droplets_numpy: expected (M, 8) = (tau, x, y, eta, E, px, py, pz)");
             const double *p = static_cast<double *>(buf.ptr);
             for (py::ssize_t i = 0; i < buf.shape[0]; ++i) {
               std::array<Jetscape::real, 4> xmu, pmu;
               for (int k = 0; k < 4; ++k) {
                 xmu[k] = static_cast<Jetscape::real>(p[i * 8 + k]);
                 pmu[k] = static_cast<Jetscape::real>(p[i * 8 + 4 + k]);
               }
               liq.add_a_droplet(Droplet(xmu, pmu));
             }
             return (int)buf.shape[0];
           },
           "Append droplets from an (M, 8) array in the droplets_numpy() column order. "
           "Used by the replay path.",
           py::arg("arr"))
      .def("get_source",
           [](const LiquefierBase &liq, Jetscape::real tau, Jetscape::real x,
              Jetscape::real y, Jetscape::real eta) {
             std::array<Jetscape::real, 4> jmu = {0.0, 0.0, 0.0, 0.0};
             liq.get_source(tau, x, y, eta, jmu);
             return jmu;
           },
           "Summed source current j^mu [GeV/fm^4] at a Milne point, causality-gated. "
           "This is the reference the pure-Python kernel is checked against.",
           py::arg("tau"), py::arg("x"), py::arg("y"), py::arg("eta"));

  // ── CausalLiquefier ───────────────────────────────────────────────────────
  py::class_<CausalLiquefier, LiquefierBase, std::shared_ptr<CausalLiquefier>>(
      m, "CausalLiquefier",
      "Causal (telegraph) diffusion of deposited energy-momentum.\n"
      "The default constructor reads <Liquefier><CausalLiquefier> from the loaded JETSCAPE "
      "XML; the 4-argument one takes the grid directly and needs no XML.")
      .def(py::init<>())
      .def(py::init<double, double, double, double>(),
           py::arg("dtau"), py::arg("dx"), py::arg("dy"), py::arg("deta"))
      .def_readwrite("dtau", &CausalLiquefier::dtau)
      .def_readwrite("dx", &CausalLiquefier::dx)
      .def_readwrite("dy", &CausalLiquefier::dy)
      .def_readwrite("deta", &CausalLiquefier::deta)
      .def_readwrite("tau_delay", &CausalLiquefier::tau_delay)
      .def_readwrite("time_relax", &CausalLiquefier::time_relax)
      .def_readwrite("d_diff", &CausalLiquefier::d_diff)
      .def_readwrite("width_delta", &CausalLiquefier::width_delta)
      .def_readonly("c_diff", &CausalLiquefier::c_diff)
      .def_readonly("gamma_relax", &CausalLiquefier::gamma_relax)
      // The kernel pieces, so the pure-Python port can be checked term by term against the
      // very C++ a run uses (rather than against a transcribed unit test).
      .def("kernel_rho", &CausalLiquefier::kernel_rho, py::arg("t"), py::arg("r"))
      .def("kernel_j", &CausalLiquefier::kernel_j, py::arg("t"), py::arg("r"))
      .def("rho_smooth", &CausalLiquefier::rho_smooth, py::arg("t"), py::arg("r"))
      .def("rho_delta", &CausalLiquefier::rho_delta, py::arg("t"), py::arg("r"))
      .def("j_smooth", &CausalLiquefier::j_smooth, py::arg("t"), py::arg("r"))
      .def("j_delta", &CausalLiquefier::j_delta, py::arg("t"), py::arg("r"))
      .def("get_ptau", &CausalLiquefier::get_ptau, py::arg("px"), py::arg("pz"), py::arg("eta"))
      .def("get_peta", &CausalLiquefier::get_peta, py::arg("px"), py::arg("pz"), py::arg("eta"))
      .def("set_t_delay", &CausalLiquefier::set_t_delay, py::arg("tau_delay"))
      .def("InitializeParameters", &CausalLiquefier::InitializeParameters,
           "Re-read the XML block and recompute c_diff / gamma_relax.")
      .def("params",
           [](const CausalLiquefier &l) {
             py::dict d;
             d["dtau"] = l.dtau;
             d["tau_delay"] = l.tau_delay;
             d["time_relax"] = l.time_relax;
             d["d_diff"] = l.d_diff;
             d["width_delta"] = l.width_delta;
             return d;
           },
           "The five parameters fast_data.liquefier.LiquefierParams needs, read from the "
           "live C++ object rather than duplicated on the Python side.");
}
