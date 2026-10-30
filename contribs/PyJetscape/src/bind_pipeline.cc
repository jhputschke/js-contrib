/*******************************************************************************
 * Copyright (c) The JETSCAPE Collaboration, 2018
 *
 * What a pipeline built in Python needs beyond the stage modules
 * (docs/Plans/PLAN_pyjetscape_complete.md, phase 2):
 *
 *   PartonShowerGenerator, PartonShowerGeneratorDefault, IsrShowerPSG
 *       how a JetEnergyLoss task grows its shower (subclassable in Python)
 *   IsrManager, IsrJet, iMATTER
 *       the initial-state radiation chain the XML builds under PythiaIsrGun
 *       (<Hard><ISR>): IsrManager > IsrJet (+ IsrShowerPSG) > iMATTER
 *   JetScapeReaderAscii, JetScapeReaderAsciiGZ
 *       read a JetScapeWriterAscii(GZ) file back, event by event, as objects
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>

#include "IsrJet.h"
#include "IsrManager.h"
#include "IsrShowerPSG.h"
#include "JetEnergyLoss.h"
#include "JetScapeReader.h"
#include "PartonShowerGenerator.h"
#include "PartonShowerGeneratorDefault.h"
#include "iMATTER.h"
#include "pyjs_modules.h"

namespace py = pybind11;
using namespace Jetscape;

namespace {

class PyPartonShowerGenerator : public PartonShowerGenerator,
                                public py::trampoline_self_life_support {
public:
  using PartonShowerGenerator::PartonShowerGenerator;

  void DoShower(JetEnergyLoss &j) override {
    PYBIND11_OVERRIDE(void, PartonShowerGenerator, DoShower, j);
  }
  void DoCalculateTime(JetEnergyLoss &j) override {
    PYBIND11_OVERRIDE(void, PartonShowerGenerator, DoCalculateTime, j);
  }
  void DoExecTime(JetEnergyLoss &j) override {
    PYBIND11_OVERRIDE(void, PartonShowerGenerator, DoExecTime, j);
  }
  void DoInitPerEvent(JetEnergyLoss &j) override {
    PYBIND11_OVERRIDE(void, PartonShowerGenerator, DoInitPerEvent, j);
  }
  void DoFinishPerEvent(JetEnergyLoss &j) override {
    PYBIND11_OVERRIDE(void, PartonShowerGenerator, DoFinishPerEvent, j);
  }
};

template <class Reader>
void bind_reader(py::module_ &m, const char *name, const char *doc) {
  py::class_<Reader, std::shared_ptr<Reader>>(m, name, doc)
      .def(py::init<std::string>(), py::arg("file_name"))
      .def("Next", &Reader::Next, "Read the next event.")
      .def("Finished", &Reader::Finished, "True at the end of the file.")
      .def("Close", &Reader::Close)
      .def("GetCurrentEvent", &Reader::GetCurrentEvent)
      .def("GetCurrentNumberOfPartonShowers",
           &Reader::GetCurrentNumberOfPartonShowers)
      .def("GetPartonShowers", &Reader::GetPartonShowers)
      .def("GetHadrons",
           [](Reader &r) { return pyjs::ptrs(r.GetHadrons()); })
      .def("GetSigmaGen", &Reader::GetSigmaGen)
      .def("GetSigmaErr", &Reader::GetSigmaErr)
      .def("GetEventWeight", &Reader::GetEventWeight)
      .def("GetEventPlaneAngle", &Reader::GetEventPlaneAngle)
      .def("GetVertexX", &Reader::GetVertexX)
      .def("GetVertexY", &Reader::GetVertexY)
      .def("GetVertexZ", &Reader::GetVertexZ);
}

} // namespace

void bind_pipeline(py::module_ &m) {

  // ── shower generators ─────────────────────────────────────────────────────
  py::class_<PartonShowerGenerator, PyPartonShowerGenerator, py::smart_holder>(
      m, "PartonShowerGenerator",
      "How a JetEnergyLoss task grows its shower (DoShower(jloss), and the per-time-step "
      "hooks). Subclass it in Python and hand it to JetEnergyLoss."
      "AddPartonShowerGenerator().")
      .def(py::init<>())
      .def("DoShower", &PartonShowerGenerator::DoShower, py::arg("jloss"))
      .def("DoCalculateTime", &PartonShowerGenerator::DoCalculateTime,
           py::arg("jloss"))
      .def("DoExecTime", &PartonShowerGenerator::DoExecTime, py::arg("jloss"))
      .def("DoInitPerEvent", &PartonShowerGenerator::DoInitPerEvent,
           py::arg("jloss"))
      .def("DoFinishPerEvent", &PartonShowerGenerator::DoFinishPerEvent,
           py::arg("jloss"));
  py::class_<PartonShowerGeneratorDefault, PartonShowerGenerator,
             py::smart_holder>(m, "PartonShowerGeneratorDefault",
                               "The final-state shower generator every "
                               "JetEnergyLoss task uses by default.")
      .def(py::init<>());
  py::class_<IsrShowerPSG, PartonShowerGeneratorDefault, py::smart_holder>(
      m, "IsrShowerPSG", "The initial-state (backward) shower generator of IsrJet.")
      .def(py::init<>());

  // ── the ISR chain ─────────────────────────────────────────────────────────
  py::class_<IsrManager, JetEnergyLossManager, py::smart_holder>(
      m, "IsrManager",
      R"pbdoc(
        The initial-state radiation of the hard partons: an energy-loss manager run
        backward in time under PythiaIsrGun. As the XML builds it for <Hard><ISR>:

            isr = IsrManager(); jet = IsrJet(); imatter = iMATTER()
            jet.SetDeltaT(-dt); jet.SetStartT(0); jet.SetMaxT(-maxT); imatter.SetMaxT(-maxT)
            jet.AddPartonShowerGenerator(IsrShowerPSG()); jet.Add(imatter)
            isr.Add(jet); gun = create_module("PythiaIsrGun"); gun.Add(isr)
            gun.Add(create_module("MCGlauberGenString"))
      )pbdoc")
      .def(py::init<>());
  py::class_<IsrJet, JetEnergyLoss, py::smart_holder>(
      m, "IsrJet", "The per-parton task of the IsrManager (an initial-state shower).")
      .def(py::init<>());
  py::class_<iMATTER, JetEnergyLoss, py::smart_holder>(
      m, "iMATTER", "MATTER's initial-state (space-like) shower.")
      .def(py::init<>());

  // ── readers ───────────────────────────────────────────────────────────────
  bind_reader<JetScapeReaderAscii>(
      m, "JetScapeReaderAscii",
      "Read a JetScapeWriterAscii file: while not r.Finished(): r.Next(); "
      "r.GetPartonShowers(), r.GetHadrons(), r.GetSigmaGen(), ...");
#ifdef USE_GZIP
  bind_reader<JetScapeReaderAsciiGZ>(m, "JetScapeReaderAsciiGZ",
                                     "As JetScapeReaderAscii, for .gz files.");
  m.attr("HAS_GZIP_READER") = true;
#else
  m.attr("HAS_GZIP_READER") = false;
#endif
}
