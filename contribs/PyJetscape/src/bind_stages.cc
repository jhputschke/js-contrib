/*******************************************************************************
 * Copyright (c) The JETSCAPE Collaboration, 2018
 *
 * The module classes of the stages after the medium, each subclassable in
 * Python (docs/Plans/PLAN_pyjetscape_complete.md, phase 1):
 *
 *   HardProcess          a gun: ExecuteTask() fills the event with AddParton()
 *   Hadronization        a jet hadronization module: DoHadronization(); also the
 *                        container a HadronizationManager runs
 *   HadronizationManager the jet hadronization of an event (Mode B pipelines)
 *   SoftParticlization   a particlization of the hydro surface: ExecuteTask() fills
 *                        the event's hadron samples
 *   Afterburner          hadronic transport: ExecuteTask() and GetCurrentHadronList()
 *   JetScapeWriter       a writer: Write(obj) gets every parton, vertex, shower,
 *                        hadron and string the tasks write, WriteEvent() ends an event
 *
 * The C++ names and argument order are kept (D3); containers passed by
 * reference reach Python as lists and are copied back after the call.
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

#include "FluidCellInfo.h"
#include "Afterburner.h"
#include "Hadronization.h"
#include "HadronizationManager.h"
#include "HardProcess.h"
#include "JetClass.h"
#include "JetScapeEventHeader.h"
#include "JetScapeParticles.h"
#include "JetScapeWriter.h"
#include "PartonShower.h"
#include "SoftParticlization.h"
#include "SurfaceCellInfo.h"
#include "pyjs_modules.h"

namespace py = pybind11;
using namespace Jetscape;

namespace {

// ── HardProcess ─────────────────────────────────────────────────────────────
class PyHardProcess : public HardProcess,
                      public py::trampoline_self_life_support {
public:
  using HardProcess::HardProcess;

  void InitTask() override { PYBIND11_OVERRIDE(void, HardProcess, InitTask); }
  void ExecuteTask() override {
    PYBIND11_OVERRIDE(void, HardProcess, ExecuteTask);
  }
  void ClearTask() override { PYBIND11_OVERRIDE(void, HardProcess, ClearTask); }
  void FinishTask() override { PYBIND11_OVERRIDE(void, HardProcess, FinishTask); }
  double GetSigmaGen() override {
    PYBIND11_OVERRIDE(double, HardProcess, GetSigmaGen);
  }
  double GetSigmaErr() override {
    PYBIND11_OVERRIDE(double, HardProcess, GetSigmaErr);
  }
  double GetPtHat() override { PYBIND11_OVERRIDE(double, HardProcess, GetPtHat); }
  double GetEventWeight() override {
    PYBIND11_OVERRIDE(double, HardProcess, GetEventWeight);
  }
};

// ── Hadronization ───────────────────────────────────────────────────────────
// A module: DoHadronization(shower, hOut, pOut). shower: one list of partons per
// shower (the C++ objects, shared); hOut / pOut: the hadrons and the partons left
// over, lists the module appends to (they may hold what an earlier module made).
// The manager does not copy jet hadronization modules, so Clone() only reaches a
// Python override.
class PyHadronization : public Hadronization,
                        public py::trampoline_self_life_support {
public:
  using Hadronization::Hadronization;

  std::shared_ptr<Hadronization> Clone() const override {
    PYBIND11_OVERRIDE(std::shared_ptr<Hadronization>, Hadronization, Clone);
  }

  void DoHadronization(std::vector<std::vector<std::shared_ptr<Parton>>> &shower,
                       std::vector<std::shared_ptr<Hadron>> &hOut,
                       std::vector<std::shared_ptr<Parton>> &pOut) override {
    py::gil_scoped_acquire gil;
    py::function ov = py::get_override(static_cast<const Hadronization *>(this),
                                       "DoHadronization");
    if (!ov)
      return;
    py::list in;
    for (auto &s : shower)
      in.append(pyjs::ptrs(s));
    py::list h = pyjs::ptrs(hOut);
    py::list p = pyjs::ptrs(pOut);
    ov(in, h, p);
    std::vector<std::vector<std::shared_ptr<Parton>>> nin;
    for (auto s : in) {
      std::vector<std::shared_ptr<Parton>> v;
      pyjs::assign_ptrs(s.cast<py::list>(), v);
      nin.push_back(std::move(v));
    }
    shower.swap(nin);
    pyjs::assign_ptrs(h, hOut);
    pyjs::assign_ptrs(p, pOut);
  }

  // Hadronization::InitTask is the container's (it refuses a task without
  // modules); a Python subclass is a module, so without an override: nothing.
  void InitTask() override {
    py::gil_scoped_acquire gil;
    py::function ov =
        py::get_override(static_cast<const Hadronization *>(this), "InitTask");
    if (ov)
      ov();
  }
  void ClearTask() override {
    PYBIND11_OVERRIDE(void, Hadronization, ClearTask);
  }
  void FinishTask() override {
    PYBIND11_OVERRIDE(void, Hadronization, FinishTask);
  }
};

// ── SoftParticlization ──────────────────────────────────────────────────────
class PySoftParticlization : public SoftParticlization,
                             public py::trampoline_self_life_support {
public:
  using SoftParticlization::SoftParticlization;

  void InitTask() override {
    PYBIND11_OVERRIDE(void, SoftParticlization, InitTask);
  }
  void ExecuteTask() override {
    PYBIND11_OVERRIDE(void, SoftParticlization, ExecuteTask);
  }
  void ClearTask() override {
    PYBIND11_OVERRIDE(void, SoftParticlization, ClearTask);
  }
  void FinishTask() override {
    PYBIND11_OVERRIDE(void, SoftParticlization, FinishTask);
  }
  bool SetNumberOfSamples(int n) override {
    PYBIND11_OVERRIDE(bool, SoftParticlization, SetNumberOfSamples, n);
  }
};

// ── Afterburner ─────────────────────────────────────────────────────────────
class PyAfterburner : public Afterburner,
                      public py::trampoline_self_life_support {
public:
  using Afterburner::Afterburner;

  void InitTask() override { PYBIND11_OVERRIDE(void, Afterburner, InitTask); }
  void ExecuteTask() override {
    PYBIND11_OVERRIDE(void, Afterburner, ExecuteTask);
  }
  void ClearTask() override { PYBIND11_OVERRIDE(void, Afterburner, ClearTask); }
  void FinishTask() override { PYBIND11_OVERRIDE(void, Afterburner, FinishTask); }
  std::vector<Hadron> GetCurrentHadronList() const override {
    py::gil_scoped_acquire gil;
    py::function ov = py::get_override(static_cast<const Afterburner *>(this),
                                       "GetCurrentHadronList");
    if (!ov)
      py::pybind11_fail("Afterburner.GetCurrentHadronList() is not implemented "
                        "by the Python subclass");
    std::vector<Hadron> out;
    for (auto h : ov())
      out.push_back(h.cast<Hadron>());
    return out;
  }
};

// The protected helpers of Afterburner, reached through a using-declaration
// (the member pointers stay Afterburner's).
struct AfterburnerPublicist : Afterburner {
  using Afterburner::GatherAfterburnerHadrons;
  using Afterburner::GetFragmentationHadrons;
  using Afterburner::GetSoftParticlizationHadrons;
};

py::list hadron_samples(const std::vector<std::vector<std::shared_ptr<Hadron>>> &v) {
  py::list out;
  for (auto &s : v)
    out.append(pyjs::ptrs(s));
  return out;
}

// ── JetScapeWriter ──────────────────────────────────────────────────────────
// One Python Write(obj) receives what the C++ overloads get: Parton, Vertex,
// PartonShower, Hadron and str (Jet and Qvector are not bound and stay no-ops).
class PyJetScapeWriter : public JetScapeWriter,
                         public py::trampoline_self_life_support {
public:
  using JetScapeWriter::JetScapeWriter;

  bool GetStatus() override {
    PYBIND11_OVERRIDE_PURE(bool, JetScapeWriter, GetStatus);
  }
  void Open() override { PYBIND11_OVERRIDE(void, JetScapeWriter, Open); }
  void Close() override { PYBIND11_OVERRIDE(void, JetScapeWriter, Close); }
  void WriteInitFileXML() override {
    PYBIND11_OVERRIDE(void, JetScapeWriter, WriteInitFileXML);
  }
  void WriteHeaderToFile() override {
    PYBIND11_OVERRIDE(void, JetScapeWriter, WriteHeaderToFile);
  }
  void WriteEvent() override {
    PYBIND11_OVERRIDE(void, JetScapeWriter, WriteEvent);
  }
  void WriteComment(std::string s) override {
    PYBIND11_OVERRIDE(void, JetScapeWriter, WriteComment, s);
  }
  void WriteWhiteSpace(std::string s) override {
    PYBIND11_OVERRIDE(void, JetScapeWriter, WriteWhiteSpace, s);
  }
  void InitTask() override { PYBIND11_OVERRIDE(void, JetScapeWriter, InitTask); }
  void ExecuteTask() override {
    PYBIND11_OVERRIDE(void, JetScapeWriter, ExecuteTask);
  }
  void FinishTask() override {
    PYBIND11_OVERRIDE(void, JetScapeWriter, FinishTask);
  }

  void Write(std::weak_ptr<Parton> p) override { dispatch(p.lock()); }
  void Write(std::weak_ptr<Vertex> v) override { dispatch(v.lock()); }
  void Write(std::weak_ptr<PartonShower> ps) override { dispatch(ps.lock()); }
  void Write(std::weak_ptr<Hadron> h) override { dispatch(h.lock()); }
  void Write(std::string s) override {
    py::gil_scoped_acquire gil;
    if (py::function ov = py::get_override(static_cast<const JetScapeWriter *>(this),
                                           "Write"))
      ov(s);
  }
  using JetScapeWriter::Write;

private:
  template <class T> void dispatch(const std::shared_ptr<T> &obj) {
    if (!obj)
      return;
    py::gil_scoped_acquire gil;
    if (py::function ov = py::get_override(static_cast<const JetScapeWriter *>(this),
                                           "Write"))
      ov(obj);
  }
};

template <class Signal>
py::list hydro_hypersurface(Signal &sig, bool connected, double T) {
  py::list out;
  if (!connected)
    return out;
  std::vector<SurfaceCellInfo> cells;
  sig(T, cells);
  for (auto &c : cells)
    out.append(py::cast(c));
  return out;
}

} // namespace

void bind_stages(py::module_ &m) {

  // ── HardProcess ───────────────────────────────────────────────────────────
  py::class_<HardProcess, JetScapeModuleBase, PyHardProcess, py::smart_holder>(
      m, "HardProcess",
      R"pbdoc(
        The hard process of an event (PythiaGun, PGun, ... in C++).

        A Python gun subclasses it and fills the event in ExecuteTask() with
        AddParton(Parton(...)) (and AddHadron() for hadrons from the hard
        process); GetSigmaGen, GetSigmaErr, GetPtHat and GetEventWeight can be
        overridden for the writers. The hard vertex comes from the initial state:
        JetScapeSignalManager.Instance().GetInitialStatePointer()
        .sample_binary_collision_point(). register_module("CustomModuleX", cls)
        makes it available to the XML (<Hard><CustomModuleX>).
      )pbdoc")
      .def(py::init<>())
      .def("AddParton", &HardProcess::AddParton, py::arg("parton"))
      .def("GetNHardPartons", &HardProcess::GetNHardPartons)
      .def("GetPartonAt", &HardProcess::GetPartonAt, py::arg("i"))
      .def("GetPartonList",
           [](HardProcess &h) { return pyjs::ptrs(h.GetPartonList()); },
           "The hard partons of the event (the objects, not copies).")
      .def("AddHadron", &HardProcess::AddHadron, py::arg("hadron"))
      .def("GetNHadrons", &HardProcess::GetNHadrons)
      .def("GetHadronList",
           [](HardProcess &h) { return pyjs::ptrs(h.GetHadronList()); })
      .def("GetNPartonShowers", &HardProcess::GetNPartonShowers)
      .def("GetPartonShowerAt", &HardProcess::GetPartonShowerAt, py::arg("i"))
      .def("AddPartonShower", &HardProcess::AddPartonShower, py::arg("shower"))
      .def("GetRemnants", &HardProcess::GetRemnants,
           "The beam remnants (copies), with ISR.")
      .def("PushRemnants", &HardProcess::PushRemnants, py::arg("parton"))
      .def("GetSigmaGen", &HardProcess::GetSigmaGen)
      .def("GetSigmaErr", &HardProcess::GetSigmaErr)
      .def("GetPtHat", &HardProcess::GetPtHat)
      .def("GetEventWeight", &HardProcess::GetEventWeight)
      .def("GetMax_ColorPerShower", &HardProcess::GetMax_ColorPerShower)
      .def("GetMax_Color", &HardProcess::GetMax_Color)
      .def("SetMax_Color", &HardProcess::SetMax_Color, py::arg("col"));

  // ── Hadronization ─────────────────────────────────────────────────────────
  py::class_<Hadronization, JetScapeModuleBase, PyHadronization,
             py::smart_holder>(
      m, "Hadronization",
      R"pbdoc(
        A jet hadronization task: the container a HadronizationManager runs (Add()
        the modules to it), and the base of every jet hadronization module, C++
        (ColorlessHadronization, iColoredHadronization, ...) or Python.

        A Python module subclasses it and overrides

            def DoHadronization(self, shower, hOut, pOut): ...

        shower: one list of Partons per shower (the C++ objects); hOut: append the
        Hadrons; pOut: append the partons not hadronized. GetHydroHyperSurface(T)
        and GetHydroCell(t, x, y, z) give the medium. register_module(
        "CustomModuleX", cls) and <JetHadronization><name>CustomModuleX</name> make
        the XML use it.
      )pbdoc")
      .def(py::init<>())
      .def("DoHadronization",
           [](Hadronization &h, py::list shower, py::list hOut, py::list pOut) {
             std::vector<std::vector<std::shared_ptr<Parton>>> in;
             for (auto s : shower) {
               std::vector<std::shared_ptr<Parton>> v;
               pyjs::assign_ptrs(s.cast<py::list>(), v);
               in.push_back(std::move(v));
             }
             std::vector<std::shared_ptr<Hadron>> ho;
             std::vector<std::shared_ptr<Parton>> po;
             pyjs::assign_ptrs(hOut, ho);
             pyjs::assign_ptrs(pOut, po);
             h.DoHadronization(in, ho, po);
             hOut.attr("clear")();
             for (auto &x : ho) hOut.append(py::cast(x));
             pOut.attr("clear")();
             for (auto &x : po) pOut.append(py::cast(x));
           },
           "Hadronize: shower (list of lists of Parton), hOut and pOut (lists, "
           "filled with the hadrons and the partons left over).",
           py::arg("shower"), py::arg("hOut"), py::arg("pOut"))
      .def("GetHadrons",
           [](Hadronization &h) { return pyjs::ptrs(h.GetHadrons()); },
           "The hadrons this task made in the event.")
      .def("GetOutPartons",
           [](Hadronization &h) { return pyjs::ptrs(h.GetOutPartons()); },
           "The partons this task left over in the event.")
      .def("GetHydroHyperSurface",
           [](Hadronization &h, double T) {
             return hydro_hypersurface(h.GetHydroHyperSurface,
                                       h.GetGetHydroHyperSurfaceConnected(), T);
           },
           "The hydro's constant-temperature surface at T [GeV] as SurfaceCellInfos "
           "(empty without a hydro).",
           py::arg("T"))
      .def("GetHydroCell",
           [](Hadronization &h, double t, double x, double y,
              double z) -> py::object {
             if (!h.GetGetHydroCellSignalConnected())
               return py::none();
             std::unique_ptr<FluidCellInfo> cell;
             h.GetHydroCellSignal(t, x, y, z, cell);
             if (!cell)
               return py::none();
             return py::cast(FluidCellInfo(*cell));
           },
           py::arg("t"), py::arg("x"), py::arg("y"), py::arg("z"))
      .def("DeleteHadrons", &Hadronization::DeleteHadrons)
      .def("DeleteRealHadrons", &Hadronization::DeleteRealHadrons);

  // ── HadronizationManager ──────────────────────────────────────────────────
  py::class_<HadronizationManager, JetScapeModuleBase, py::smart_holder>(
      m, "HadronizationManager",
      "The jet hadronization of an event: it takes the energy-loss manager's final "
      "partons and runs its Hadronization tasks (Add() a Hadronization with the "
      "modules in it).")
      .def(py::init<>())
      .def("GetHadrons",
           [](HadronizationManager &hm) {
             std::vector<std::shared_ptr<Hadron>> out;
             hm.GetHadrons(out);
             return pyjs::ptrs(out);
           },
           "The hadrons of all its tasks in this event.")
      .def("DeleteHadrons", &HadronizationManager::DeleteHadrons)
      .def("DeleteRealHadrons", &HadronizationManager::DeleteRealHadrons);

  // ── SoftParticlization ────────────────────────────────────────────────────
  py::class_<SoftParticlization, JetScapeModuleBase, PySoftParticlization,
             py::smart_holder>(
      m, "SoftParticlization",
      R"pbdoc(
        Particlization of the hydro's freeze-out surface (iSS in C++).

        A Python sampler subclasses it, gets the surface in ExecuteTask() with
        GetHydroHyperSurface() (or FindHydroHyperSurface() from the stored
        evolution) and stores the event's hadrons with set_hadron_list(samples),
        one list of Hadrons per oversample. register_module("CustomModuleX", cls)
        and <SoftParticlization><CustomModuleX> make the XML use it.
      )pbdoc")
      .def(py::init<>())
      .def("get_hadron_list",
           [](SoftParticlization &sp) { return hadron_samples(sp.Hadron_list_); },
           "The event's hadrons: one list of Hadrons per oversample.")
      .def("set_hadron_list",
           [](SoftParticlization &sp, py::list samples) {
             std::vector<std::vector<std::shared_ptr<Hadron>>> v;
             for (auto s : samples) {
               std::vector<std::shared_ptr<Hadron>> one;
               pyjs::assign_ptrs(s.cast<py::list>(), one);
               v.push_back(std::move(one));
             }
             sp.Hadron_list_.swap(v);
           },
           "Store the event's hadrons: a list of oversamples, each a list of Hadrons.",
           py::arg("samples"))
      .def("GetHydroHyperSurface",
           [](SoftParticlization &sp) {
             py::list out;
             if (!sp.GetGetHydroHyperSurfaceConnected())
               return out;
             std::vector<SurfaceCellInfo> cells;
             sp.GetHydroHyperSurface(cells);
             for (auto &c : cells)
               out.append(py::cast(c));
             return out;
           },
           "The hydro's freeze-out surface (its own, as MUSIC gives it) as "
           "SurfaceCellInfos; empty without one.")
      .def("ClearHydroHyperSurface",
           [](SoftParticlization &sp) {
             if (sp.GetClearHydroHyperSurfaceConnected())
               sp.ClearHydroHyperSurface();
           })
      .def("SetNumberOfSamples", &SoftParticlization::SetNumberOfSamples,
           py::arg("n"))
      .def("SetNextRandomSeed", &SoftParticlization::SetNextRandomSeed,
           py::arg("seed"))
      .def("GetLastRandomSeed", &SoftParticlization::GetLastRandomSeed);

  // ── Afterburner ───────────────────────────────────────────────────────────
  py::class_<Afterburner, JetScapeModuleBase, PyAfterburner, py::smart_holder>(
      m, "Afterburner",
      R"pbdoc(
        Hadronic transport after particlization (SMASH in C++).

        A Python afterburner subclasses it, takes its input in ExecuteTask() with
        GatherAfterburnerHadrons() (one list per oversample: the soft hadrons plus
        the jet hadrons), propagates them, and returns the result from
        GetCurrentHadronList(). register_module("CustomModuleX", cls) and
        <Afterburner><CustomModuleX> make the XML use it.
      )pbdoc")
      .def(py::init<>())
      .def("GetCurrentHadronList", &Afterburner::GetCurrentHadronList)
      .def("GatherAfterburnerHadrons",
           [](Afterburner &a) {
             return hadron_samples(
                 (a.*(&AfterburnerPublicist::GatherAfterburnerHadrons))());
           },
           "The hadrons for the transport: per oversample, the soft and the jet "
           "hadrons.")
      .def("GetSoftParticlizationHadrons",
           [](Afterburner &a) {
             return hadron_samples(
                 (a.*(&AfterburnerPublicist::GetSoftParticlizationHadrons))());
           })
      .def("GetFragmentationHadrons", [](Afterburner &a) {
        return pyjs::ptrs((a.*(&AfterburnerPublicist::GetFragmentationHadrons))());
      });

  // ── JetScapeEventHeader ───────────────────────────────────────────────────
  py::class_<JetScapeEventHeader>(m, "JetScapeEventHeader",
      "What the tasks report for an event (CollectHeader): weights, vertex, "
      "centrality.")
#define HEADER_FIELD(name)                                                    \
  .def("Get" #name, &JetScapeEventHeader::Get##name)                          \
      .def("Set" #name, &JetScapeEventHeader::Set##name, py::arg("value"))
      HEADER_FIELD(SigmaGen) HEADER_FIELD(SigmaErr) HEADER_FIELD(PtHat)
      HEADER_FIELD(EventWeight) HEADER_FIELD(VertexX) HEADER_FIELD(VertexY)
      HEADER_FIELD(VertexZ) HEADER_FIELD(Npart) HEADER_FIELD(Ncoll)
      HEADER_FIELD(EventCentrality) HEADER_FIELD(TotalEntropy)
      HEADER_FIELD(EventPlaneAngle)
#undef HEADER_FIELD
      ;

  // ── JetScapeWriter ────────────────────────────────────────────────────────
  py::class_<JetScapeWriter, JetScapeModuleBase, PyJetScapeWriter,
             py::smart_holder>(
      m, "JetScapeWriter",
      R"pbdoc(
        An output writer (ASCII, HepMC, ... in C++).

        A Python writer subclasses it, implements GetStatus() (True when ready),
        and overrides Write(obj): it gets every Parton, Vertex, PartonShower,
        Hadron and str the tasks write in an event, after WriteHeaderToFile()
        (GetHeader() then holds the event header) and before WriteEvent(). Add
        it to the pipeline with JetScape.Add(writer).
      )pbdoc")
      .def(py::init<>())
      .def(py::init<std::string>(), py::arg("file_name"))
      .def("SetOutputFileName", &JetScapeWriter::SetOutputFileName,
           py::arg("file_name"))
      .def("GetOutputFileName", &JetScapeWriter::GetOutputFileName)
      .def("GetStatus", &JetScapeWriter::GetStatus)
      .def("Open", &JetScapeWriter::Open)
      .def("Close", &JetScapeWriter::Close)
      .def("WriteHeaderToFile", &JetScapeWriter::WriteHeaderToFile)
      .def("WriteEvent", &JetScapeWriter::WriteEvent)
      .def("WriteComment", &JetScapeWriter::WriteComment, py::arg("s"))
      .def("WriteWhiteSpace", &JetScapeWriter::WriteWhiteSpace, py::arg("s"))
      .def("Write",
           [](JetScapeWriter &w, std::shared_ptr<Parton> p) {
             w.Write(std::weak_ptr<Parton>(p));
           },
           py::arg("obj"))
      .def("Write",
           [](JetScapeWriter &w, std::shared_ptr<Hadron> h) {
             w.Write(std::weak_ptr<Hadron>(h));
           },
           py::arg("obj"))
      .def("Write",
           [](JetScapeWriter &w, std::shared_ptr<Vertex> v) {
             w.Write(std::weak_ptr<Vertex>(v));
           },
           py::arg("obj"))
      .def("Write",
           [](JetScapeWriter &w, std::shared_ptr<PartonShower> ps) {
             w.Write(std::weak_ptr<PartonShower>(ps));
           },
           py::arg("obj"))
      .def("Write", [](JetScapeWriter &w, const std::string &s) { w.Write(s); },
           py::arg("obj"))
      .def("GetHeader", &JetScapeWriter::GetHeader,
           py::return_value_policy::reference_internal);
}
