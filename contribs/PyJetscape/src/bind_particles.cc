/*******************************************************************************
 * Copyright (c) The JETSCAPE Collaboration, 2018
 *
 * pybind11 bindings for the particle classes: FourVector, JetScapeParticleBase,
 * Parton, Photon and Hadron, with their constructors and setters, so a Python
 * module can make and change particles (a gun, an energy-loss step, a
 * hadronization).
 *
 * Names follow the C++ API (set_stat, set_form_time, set_color, ...), with one
 * exception kept from the first bindings: x(), y(), z(), t() of a particle are
 * its POSITION (x_in(), t in fm/c). jetscape.showers writes the shower initiators
 * with them. The virtuality, C++ Parton::t(), is Parton.virtuality(); C++'s
 * set_t() sets it.
 *
 * Holder: py::smart_holder (as the module classes), so particles made in Python
 * can be handed to C++ as shared_ptr (HardProcess::AddParton, hadron lists).
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <sstream>
#include <string>

#include "FourVector.h"
#include "JetScapeParticles.h"

namespace py = pybind11;
using namespace Jetscape;

namespace {

std::string four_repr(const FourVector &v) {
  std::ostringstream s;
  s << "FourVector(x=" << v.x() << ", y=" << v.y() << ", z=" << v.z()
    << ", t=" << v.t() << ")";
  return s.str();
}

template <class P> std::string particle_repr(const char *name, P &p) {
  std::ostringstream s;
  s << name << "(pid=" << p.pid() << ", pstat=" << p.pstat()
    << ", label=" << p.plabel() << ", E=" << p.e() << ", px=" << p.px()
    << ", py=" << p.py() << ", pz=" << p.pz() << ", x=" << p.x_in().x()
    << ", y=" << p.x_in().y() << ", z=" << p.x_in().z()
    << ", t=" << p.x_in().t() << ")";
  return s.str();
}

// C++ set_x(double[4]) takes (t, x, y, z); from a FourVector
void set_position(JetScapeParticleBase &p, const FourVector &x) {
  double a[4] = {x.t(), x.x(), x.y(), x.z()};
  p.set_x(a);
}

} // namespace

void bind_particles(py::module_ &m) {

  // ── FourVector ────────────────────────────────────────────────────────────
  py::class_<FourVector>(m, "FourVector",
      "A four-vector. The constructor takes (x, y, z, t) as C++ "
      "FourVector(x_in, y_in, z_in, t_in); momenta are (px, py, pz, E).")
      .def(py::init<>())
      .def(py::init<double, double, double, double>(), py::arg("x"),
           py::arg("y"), py::arg("z"), py::arg("t"))
      .def(py::init<const FourVector &>(), py::arg("other"))
      .def("x", &FourVector::x)
      .def("y", &FourVector::y)
      .def("z", &FourVector::z)
      .def("t", &FourVector::t)
      .def("comp", &FourVector::comp, py::arg("i"),
           "Component i: 0 = t, 1 = x, 2 = y, 3 = z.")
      .def("Set",
           [](FourVector &v, double x, double y, double z, double t) {
             v.Set(x, y, z, t);
           },
           py::arg("x"), py::arg("y"), py::arg("z"), py::arg("t"))
      .def("clear", &FourVector::clear)
      .def("plus", &FourVector::plus)
      .def("minus", &FourVector::minus)
      .def("rapidity", &FourVector::rapidity)
      .def("eta", &FourVector::eta)
      .def("phi", &FourVector::phi)
      .def("rotate_around_x", &FourVector::rotate_around_x, py::arg("theta"))
      .def("rotate_around_y", &FourVector::rotate_around_y, py::arg("theta"))
      .def("rotate_around_z", &FourVector::rotate_around_z, py::arg("theta"))
      .def("eta_boost", &FourVector::eta_boost, py::arg("deta"))
      .def("y_boost", &FourVector::y_boost, py::arg("dy"))
      .def("boost", &FourVector::boost, py::arg("vx"), py::arg("vy"),
           py::arg("vz"))
      .def("__iadd__",
           [](FourVector &a, FourVector &b) -> FourVector & { return a += b; },
           py::return_value_policy::reference_internal)
      .def("__isub__",
           [](FourVector &a, FourVector &b) -> FourVector & { return a -= b; },
           py::return_value_policy::reference_internal)
      .def("__add__",
           [](const FourVector &a, FourVector &b) {
             FourVector c(a);
             return c += b, c;
           })
      .def("__sub__",
           [](const FourVector &a, FourVector &b) {
             FourVector c(a);
             return c -= b, c;
           })
      .def("to_list", [](const FourVector &v) {
             return std::vector<double>{v.x(), v.y(), v.z(), v.t()};
           },
           "[x, y, z, t]")
      .def("__repr__", &four_repr);

  // ── JetScapeParticleBase ──────────────────────────────────────────────────
  py::class_<JetScapeParticleBase, py::smart_holder>(m, "JetScapeParticleBase",
      "Common base of Parton and Hadron: identity, momentum, position.")
      // identity
      .def("pid", &JetScapeParticleBase::pid)
      .def("pstat", &JetScapeParticleBase::pstat)
      .def("plabel", &JetScapeParticleBase::plabel)
      .def("set_id", &JetScapeParticleBase::set_id, py::arg("id"))
      .def("set_stat", &JetScapeParticleBase::set_stat, py::arg("stat"))
      .def("set_label", &JetScapeParticleBase::set_label, py::arg("label"))
      // momentum (GeV)
      .def("e", [](JetScapeParticleBase &p) { return p.e(); })
      .def("px", [](JetScapeParticleBase &p) { return p.px(); })
      .def("py", [](JetScapeParticleBase &p) { return p.py(); })
      .def("pz", [](JetScapeParticleBase &p) { return p.pz(); })
      .def("pt", [](JetScapeParticleBase &p) { return p.pt(); })
      .def("eta", [](JetScapeParticleBase &p) { return p.eta(); })
      .def("phi", [](JetScapeParticleBase &p) { return p.phi(); },
           "Azimuth in [0, 2 pi) (fastjet's phi).")
      .def("rap", [](JetScapeParticleBase &p) { return p.rap(); })
      .def("rapidity", [](JetScapeParticleBase &p) { return p.rapidity(); })
      .def("pl", &JetScapeParticleBase::pl)
      .def("nu", &JetScapeParticleBase::nu)
      .def("p", &JetScapeParticleBase::p, py::arg("i"),
           "Momentum component i: 0 = E, 1 = px, 2 = py, 3 = pz.")
      .def("p_in", &JetScapeParticleBase::p_in,
           "Momentum as FourVector(px, py, pz, E).")
      .def("reset_momentum",
           [](JetScapeParticleBase &p, double px, double py, double pz,
              double e) { p.reset_momentum(px, py, pz, e); },
           py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("e"))
      .def("reset_momentum",
           [](JetScapeParticleBase &p, const FourVector &v) {
             p.reset_momentum(v);
           },
           py::arg("p"))
      .def("restmass", &JetScapeParticleBase::restmass,
           "Rest mass [GeV] (the PDG mass the particle was built with).")
      // position (fm, fm/c)
      .def("x_in", &JetScapeParticleBase::x_in,
           "Position as FourVector(x, y, z, t).")
      .def("set_x", &set_position, py::arg("x"),
           "Set the position from a FourVector(x, y, z, t).")
      .def("time", &JetScapeParticleBase::time, "Position time x_in().t().")
      .def("x", [](JetScapeParticleBase &p) { return p.x_in().x(); })
      .def("y", [](JetScapeParticleBase &p) { return p.x_in().y(); })
      .def("z", [](JetScapeParticleBase &p) { return p.x_in().z(); })
      .def("t", [](JetScapeParticleBase &p) { return p.x_in().t(); },
           "Position time x_in().t() [fm/c]. For a Parton the virtuality is "
           "virtuality() (C++ Parton::t()).")
      // jet velocity
      .def("jet_v", &JetScapeParticleBase::jet_v)
      .def("set_jet_v",
           [](JetScapeParticleBase &p, const FourVector &v) { p.set_jet_v(v); },
           py::arg("v"))
      .def("init_jet_v", &JetScapeParticleBase::init_jet_v)
      // controller (which module owns the parton in a mutex)
      .def("SetController", &JetScapeParticleBase::SetController,
           py::arg("controller") = std::string())
      .def("UnsetController", &JetScapeParticleBase::UnsetController)
      .def("GetController", &JetScapeParticleBase::GetController)
      .def("GetControlled", &JetScapeParticleBase::GetControlled);

  // ── Parton ────────────────────────────────────────────────────────────────
  py::class_<Parton, JetScapeParticleBase, py::smart_holder>(m, "Parton",
      R"pbdoc(
        A parton: Parton(label, id, stat, p, x) with p = FourVector(px, py, pz, E)
        [GeV] and x = FourVector(x, y, z, t) [fm]; or Parton(label, id, stat, pt,
        eta, phi, e, x=None) with x a list (t, x, y, z) as C++ double[4].

        x(), y(), z(), t() are the position; the virtuality is virtuality()
        (C++ Parton::t()), set with set_t().
      )pbdoc")
      .def(py::init<int, int, int, const FourVector &, const FourVector &>(),
           py::arg("label"), py::arg("id"), py::arg("stat"), py::arg("p"),
           py::arg("x"))
      .def(py::init([](int label, int id, int stat, double pt, double eta,
                       double phi, double e, py::object x) {
             if (x.is_none())
               return std::make_shared<Parton>(label, id, stat, pt, eta, phi, e);
             auto v = x.cast<std::vector<double>>();
             if (v.size() != 4)
               throw py::value_error("Parton: x needs 4 values (t, x, y, z)");
             return std::make_shared<Parton>(label, id, stat, pt, eta, phi, e,
                                             v.data());
           }),
           py::arg("label"), py::arg("id"), py::arg("stat"), py::arg("pt"),
           py::arg("eta"), py::arg("phi"), py::arg("e"),
           py::arg("x") = py::none())
      .def(py::init<const Parton &>(), py::arg("other"), "A copy.")
      .def("copy", [](const Parton &p) { return Parton(p); }, "A copy.")
      .def("__copy__", [](const Parton &p) { return Parton(p); })
      // virtuality and formation time
      .def("virtuality", &Parton::t, "C++ Parton::t(): the virtuality [GeV^2].")
      .def("set_t", &Parton::set_t, py::arg("t"),
           "C++ Parton::set_t(): set the virtuality (rescales the momentum).")
      .def("form_time", &Parton::form_time)
      .def("set_form_time", &Parton::set_form_time, py::arg("form_time"))
      .def("mean_form_time", &Parton::mean_form_time)
      .def("set_mean_form_time", &Parton::set_mean_form_time)
      .def("reset_p", &Parton::reset_p, py::arg("px"), py::arg("py"),
           py::arg("pz"), "Set the 3-momentum, keeping E.")
      // colour
      .def("color", &Parton::color, "Colour tag (0 = none).")
      .def("anti_color", &Parton::anti_color, "Anti-colour tag (0 = none).")
      .def("max_color", &Parton::max_color)
      .def("min_color", &Parton::min_color)
      .def("min_anti_color", &Parton::min_anti_color)
      .def("set_color", &Parton::set_color, py::arg("col"))
      .def("set_anti_color", &Parton::set_anti_color, py::arg("acol"))
      .def("set_max_color", &Parton::set_max_color, py::arg("col"))
      .def("set_min_color", &Parton::set_min_color, py::arg("col"))
      .def("set_min_anti_color", &Parton::set_min_anti_color, py::arg("acol"))
      // shower bookkeeping
      .def("edgeid", &Parton::edgeid)
      .def_static("isLepton", &Parton::isLepton, py::arg("pid"))
      .def_static("isNonParton", &Parton::isNonParton, py::arg("pid"))
      .def("isPhoton", &Parton::isPhoton, py::arg("pid"))
      .def("__repr__", [](Parton &p) { return particle_repr("Parton", p); });

  // ── Photon ────────────────────────────────────────────────────────────────
  py::class_<Photon, Parton, py::smart_holder>(m, "Photon",
      "A photon from a shower: Photon(label, id, stat, p, x) as Parton.")
      .def(py::init<int, int, int, const FourVector &, const FourVector &>(),
           py::arg("label"), py::arg("id"), py::arg("stat"), py::arg("p"),
           py::arg("x"))
      .def("__repr__", [](Photon &p) { return particle_repr("Photon", p); });

  // ── Hadron ────────────────────────────────────────────────────────────────
  py::class_<Hadron, JetScapeParticleBase, py::smart_holder>(m, "Hadron",
      R"pbdoc(
        A hadron: Hadron(label, id, stat, p, x[, mass[, charge, baryon_number,
        strangeness[, participant]]]) with p = FourVector(px, py, pz, E) [GeV] and
        x = FourVector(x, y, z, t) [fm]. Without a mass the PDG mass is used.
      )pbdoc")
      .def(py::init<int, int, int, const FourVector &, const FourVector &>(),
           py::arg("label"), py::arg("id"), py::arg("stat"), py::arg("p"),
           py::arg("x"))
      .def(py::init<int, int, int, const FourVector &, const FourVector &,
                    double>(),
           py::arg("label"), py::arg("id"), py::arg("stat"), py::arg("p"),
           py::arg("x"), py::arg("mass"))
      .def(py::init<int, int, int, const FourVector &, const FourVector &,
                    double, int, int, int>(),
           py::arg("label"), py::arg("id"), py::arg("stat"), py::arg("p"),
           py::arg("x"), py::arg("mass"), py::arg("charge"),
           py::arg("baryon_number"), py::arg("strangeness"))
      .def(py::init<int, int, int, const FourVector &, const FourVector &,
                    double, int, int, int, bool>(),
           py::arg("label"), py::arg("id"), py::arg("stat"), py::arg("p"),
           py::arg("x"), py::arg("mass"), py::arg("charge"),
           py::arg("baryon_number"), py::arg("strangeness"),
           py::arg("participant"))
      .def(py::init<const Hadron &>(), py::arg("other"), "A copy.")
      .def("copy", [](const Hadron &h) { return Hadron(h); }, "A copy.")
      .def("__copy__", [](const Hadron &h) { return Hadron(h); })
      .def("charge", &Hadron::charge)
      .def("baryon_number", &Hadron::baryon_number)
      .def("strangeness", &Hadron::strangeness)
      .def("participant", &Hadron::participant)
      .def("decay_width", &Hadron::decay_width)
      .def("set_charge", &Hadron::set_charge, py::arg("charge"))
      .def("set_baryon_number", &Hadron::set_baryon_number,
           py::arg("baryon_number"))
      .def("set_strangeness", &Hadron::set_strangeness, py::arg("strangeness"))
      .def("set_participant", &Hadron::set_participant, py::arg("participant"))
      .def("set_decay_width", &Hadron::set_decay_width, py::arg("width"))
      .def("has_no_position", &Hadron::has_no_position)
      .def("has_valid_momentum", &Hadron::has_valid_momentum)
      .def("__repr__", [](Hadron &h) { return particle_repr("Hadron", h); });
}
