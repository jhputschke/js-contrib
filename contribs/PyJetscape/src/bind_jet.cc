/*******************************************************************************
 * Copyright (c) The JETSCAPE Collaboration, 2018
 *
 * pybind11 bindings for the jet sector: parton showers and the energy-loss
 * manager.  These expose, at the per-event Python yield point, the full parton
 * shower(s) produced by the JetEnergyLossManager — the same data the ASCII
 * writer serialises — so a Python consumer (e.g. the hydro+jet visualizer) can
 * read the shower live instead of parsing the output file.
 *
 * Access path (mirrors JetEnergyLoss::WriteTask):
 *     sm    = JetScapeSignalManager.Instance()
 *     jm    = sm.GetJetEnergyLossManagerPointer()      # JetEnergyLossManager
 *     for ps in jm.get_showers():                      # one per shower-initiating parton
 *         edges = ps.to_numpy()                        # (n_partons, 12)
 *
 * The full shower is only complete *after* the JetEnergyLossManager has executed
 * (its child JetEnergyLoss copies each hold a finished PartonShower).
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <memory>
#include <vector>

#include "FourVector.h"
#include "JetClass.h"               // Vertex
#include "JetScapeParticles.h"      // Parton (: JetScapeParticleBase)
#include "PartonShower.h"           // PartonShower (: GTL graph)
#include "JetEnergyLoss.h"          // JetEnergyLoss::GetShower()
#include "JetEnergyLossManager.h"
#include "JetScapeModuleBase.h"
#include "LiquefierBase.h"
#include "FluidCellInfo.h"
#include "pyjs_modules.h"

namespace py = pybind11;
using namespace Jetscape;

// Columns of JetEnergyLossManager.final_partons_numpy() (and of the jet
// hadronization helpers in bind_hadronization.cc, which read them back).
static constexpr int kFinalPartonNCols = 14;
static const char *kFinalPartonColumns[kFinalPartonNCols] = {
    "shower", "pid", "pstat", "E", "px", "py", "pz",
    "t",      "x",   "y",     "z", "mass", "col", "acol"};

// ── PyJetEnergyLoss: an energy-loss module written in Python ─────────────────
// Subclass JetEnergyLoss and override DoEnergyLoss(deltaT, time, Q2, pIn, pOut):
// pIn holds the parton(s) of this step (the C++ objects, changed in place), pOut
// is a list the module appends the partons at time + deltaT to (empty pOut: the
// parton continues unchanged, as for a C++ module). The manager copies the module
// once per shower through Clone() (pyjs::default_clone unless overridden).
class PyJetEnergyLoss : public JetEnergyLoss,
                        public py::trampoline_self_life_support {
public:
  PyJetEnergyLoss() = default;
  PyJetEnergyLoss(const JetEnergyLoss &other) : JetEnergyLoss(other) {}

  std::shared_ptr<JetEnergyLoss> Clone() const override {
    py::gil_scoped_acquire gil;
    py::function ov =
        py::get_override(static_cast<const JetEnergyLoss *>(this), "Clone");
    if (ov)
      return ov().cast<std::shared_ptr<JetEnergyLoss>>();
    return pyjs::default_clone<JetEnergyLoss>(this);
  }

  void DoEnergyLoss(double deltaT, double time, double Q2,
                    std::vector<Parton> &pIn,
                    std::vector<Parton> &pOut) override {
    py::gil_scoped_acquire gil;
    py::function ov = py::get_override(static_cast<const JetEnergyLoss *>(this),
                                       "DoEnergyLoss");
    if (!ov)
      return;
    py::list in = pyjs::refs(pIn);
    py::list out = pyjs::copies(pOut);
    ov(deltaT, time, Q2, in, out);
    pyjs::write_back(in, pIn);
    pyjs::assign(out, pOut);
  }

  // JetEnergyLoss::InitTask is the container's (it refuses a task without
  // modules); a Python subclass is a module, so without an override: nothing.
  void InitTask() override {
    py::gil_scoped_acquire gil;
    py::function ov =
        py::get_override(static_cast<const JetEnergyLoss *>(this), "InitTask");
    if (ov)
      ov();
  }
  void ClearTask() override { PYBIND11_OVERRIDE(void, JetEnergyLoss, ClearTask); }
  void FinishTask() override {
    PYBIND11_OVERRIDE(void, JetEnergyLoss, FinishTask);
  }
};

void bind_jet(py::module_ &m) {
  {
    py::tuple cols(kFinalPartonNCols);
    for (int c = 0; c < kFinalPartonNCols; ++c)
      cols[c] = py::str(kFinalPartonColumns[c]);
    m.attr("FINAL_PARTON_COLUMNS") = cols;
  }

  // Parton, Hadron and FourVector are bound in bind_particles.cc (before this file).

  // ── Vertex (a node in the shower graph) ───────────────────────────────────
  py::class_<Vertex, std::shared_ptr<Vertex>>(m, "Vertex",
      "A splitting vertex: space-time position (x,y,z) [fm], t [fm/c].")
      .def("x", [](Vertex &v) { return v.x_in().x(); })
      .def("y", [](Vertex &v) { return v.x_in().y(); })
      .def("z", [](Vertex &v) { return v.x_in().z(); })
      .def("t", [](Vertex &v) { return v.x_in().t(); });

  // ── PartonShower (GTL graph: nodes = Vertex, edges = Parton) ───────────────
  py::class_<PartonShower, std::shared_ptr<PartonShower>>(m, "PartonShower",
      "Parton shower as a directed graph; iterate edges via to_numpy().")
      .def("number_of_partons",  &PartonShower::GetNumberOfPartons)
      .def("number_of_vertices", &PartonShower::GetNumberOfVertices)
      .def("parton_at", &PartonShower::GetPartonAt, py::arg("i"),
           "Parton on the i-th edge.")
      .def("vertex_at", &PartonShower::GetVertexAt, py::arg("i"),
           "Vertex on the i-th node.")
      .def("to_numpy",
           [](PartonShower &s) -> py::array_t<double> {
             const int ne = s.GetNumberOfPartons();
             py::array_t<double> arr({ne, 12});
             auto b = arr.mutable_unchecked<2>();
             int i = 0;
             for (auto it = s.edges_begin(); it != s.edges_end() && i < ne;
                  ++it, ++i) {
               edge e = *it;
               auto p = s.GetParton(e);
               b(i, 0) = e.source().id();
               b(i, 1) = e.target().id();
               b(i, 2) = p ? p->pid()      : 0.0;
               b(i, 3) = p ? p->pstat()    : 0.0;
               b(i, 4) = p ? p->p_in().x() : 0.0;  // px
               b(i, 5) = p ? p->p_in().y() : 0.0;  // py
               b(i, 6) = p ? p->p_in().z() : 0.0;  // pz
               b(i, 7) = p ? p->p_in().t() : 0.0;  // E
               b(i, 8)  = p ? p->x_in().x() : 0.0; // x
               b(i, 9)  = p ? p->x_in().y() : 0.0; // y
               b(i, 10) = p ? p->x_in().z() : 0.0; // z
               b(i, 11) = p ? p->x_in().t() : 0.0; // t
             }
             return arr;
           },
           R"pbdoc(
             Flatten the shower's edges to a numpy float64 array, one row per
             parton, shape (n_partons, 12):

               [source_id, target_id, pid, pstat, px, py, pz, E, x, y, z, t]

             Momentum (px,py,pz,E) in GeV; position (x,y,z) in fm; t in fm/c.
             (x,y,z,t) is the parton's production position (source vertex);
             use source_id / target_id to correlate with vertices_to_numpy().
           )pbdoc")
      .def("vertices_to_numpy",
           [](PartonShower &s) -> py::array_t<double> {
             const int nv = s.GetNumberOfVertices();
             py::array_t<double> arr({nv, 5});
             auto b = arr.mutable_unchecked<2>();
             int i = 0;
             for (auto it = s.nodes_begin(); it != s.nodes_end() && i < nv;
                  ++it, ++i) {
               node n = *it;
               auto v = s.GetVertex(n);
               b(i, 0) = n.id();
               b(i, 1) = v ? v->x_in().x() : 0.0;  // x
               b(i, 2) = v ? v->x_in().y() : 0.0;  // y
               b(i, 3) = v ? v->x_in().z() : 0.0;  // z
               b(i, 4) = v ? v->x_in().t() : 0.0;  // t
             }
             return arr;
           },
           R"pbdoc(
             Flatten the shower's nodes to a numpy float64 array, one row per
             splitting vertex, shape (n_vertices, 5):

               [node_id, x, y, z, t]

             Position (x,y,z) in fm; t in fm/c.  node_id matches the
             source_id / target_id columns in to_numpy(), allowing the two
             arrays to be joined into a directed graph (see
             jetscape.utils.shower_to_networkx).
           )pbdoc");

  // ── JetEnergyLoss — the per-shower energy-loss task ───────────────────────
  // Not in the module factory (JetScape.cc:651-652 hand-constructs it), so
  // create_module("JetEnergyLoss") does not work; a hand-wired Python pipeline needs this.
  py::class_<JetEnergyLoss, JetScapeModuleBase, PyJetEnergyLoss,
             py::smart_holder>(
      m, "JetEnergyLoss",
      R"pbdoc(
        An energy-loss task: the per-shower container the manager copies (Add() the
        algorithm modules to it, Matter before LBT), and the base of every
        energy-loss module, C++ (Matter, LBT, ...) or Python.

        A Python module subclasses it and overrides

            def DoEnergyLoss(self, deltaT, time, Q2, pIn, pOut): ...

        pIn: the parton(s) of this step, the C++ objects (change them in place;
        valid during the call only). pOut: append the partons at time + deltaT
        (left empty, the parton goes on unchanged). GetHydroCell(t, x, y, z) gives
        the medium. The manager copies the module once per shower with Clone():
        by default the C++ copy constructor plus a shallow copy of the instance
        attributes; override Clone() for anything deeper.

        register_module("CustomModuleX", cls) makes it available to the XML
        (<Eloss><CustomModuleX>...</CustomModuleX>).
      )pbdoc")
      .def(py::init<>())
      .def(py::init_alias<const JetEnergyLoss &>(), py::arg("other"),
           "Copy (the C++ copy constructor: id, settings, a new task number).")
      .def("add_a_liquefier", &JetEnergyLoss::add_a_liquefier,
           "Attach the liquefier that collects droplets from this shower. The framework "
           "calls liquefier->add_hydro_sources() centrally in DoExecTime(), so Matter and "
           "LBT need no changes.",
           py::arg("liquefier"))
      .def("get_liquefier",
           [](JetEnergyLoss &j) -> std::shared_ptr<LiquefierBase> {
             return j.get_liquefier().lock();
           },
           "The attached liquefier, or None.")
      .def("GetShower", &JetEnergyLoss::GetShower,
           "The finished PartonShower for this task (valid after Exec).")
      // ── the module interface ───────────────────────────────────────────────
      .def("DoEnergyLoss",
           [](JetEnergyLoss &j, double deltaT, double time, double Q2,
              py::list pIn, py::list pOut) {
             // a C++ module called from Python (e.g. a Python module delegating
             // to Matter): copies in, results written back to the same lists
             std::vector<Parton> in, out;
             for (auto h : pIn) in.push_back(h.cast<Parton>());
             for (auto h : pOut) out.push_back(h.cast<Parton>());
             j.DoEnergyLoss(deltaT, time, Q2, in, out);
             for (size_t i = 0; i < in.size() && i < py::len(pIn); ++i)
               pIn[i].cast<Parton &>() = in[i];
             pOut.attr("clear")();
             for (auto &p : out) pOut.append(py::cast(Parton(p)));
           },
           "One step: pIn (list of Parton, changed in place), pOut (list, filled with "
           "the partons at time + deltaT).",
           py::arg("deltaT"), py::arg("time"), py::arg("Q2"), py::arg("pIn"),
           py::arg("pOut"))
      .def("Clone", &JetEnergyLoss::Clone,
           "A copy for one shower (C++ modules: their copy constructor).")
      .def("GetHydroCell",
           [](JetEnergyLoss &j, double t, double x, double y, double z) -> py::object {
             if (!j.GetGetHydroCellSignalConnected())
               return py::none();
             std::unique_ptr<FluidCellInfo> cell;
             j.GetHydroCellSignal(t, x, y, z, cell);
             if (!cell)
               return py::none();
             return py::cast(FluidCellInfo(*cell));
           },
           "The medium at (t, x, y, z) [fm] as a FluidCellInfo, or None without a hydro "
           "(the GetHydroCell signal Matter and LBT use).",
           py::arg("t"), py::arg("x"), py::arg("y"), py::arg("z"))
      .def("GetQhat", &JetEnergyLoss::GetQhat)
      .def("SetQhat", &JetEnergyLoss::SetQhat, py::arg("qhat"))
      .def("GetDeltaT", &JetEnergyLoss::GetDeltaT)
      .def("SetDeltaT", &JetEnergyLoss::SetDeltaT, py::arg("deltaT"))
      .def("GetMaxT", &JetEnergyLoss::GetMaxT)
      .def("SetMaxT", &JetEnergyLoss::SetMaxT, py::arg("maxT"))
      .def("GetStartT", &JetEnergyLoss::GetStartT)
      .def("SetStartT", &JetEnergyLoss::SetStartT, py::arg("startT"))
      .def("GetShowerInitiatingParton", &JetEnergyLoss::GetShowerInitiatingParton,
           "The parton this shower started from (the per-shower copies).")
      .def("GetGetHydroCellSignalConnected",
           &JetEnergyLoss::GetGetHydroCellSignalConnected)
      .def("GetSentInPartonsConnected", &JetEnergyLoss::GetSentInPartonsConnected);

  // ── JetEnergyLossManager — per-event access to the finished showers ───────
  // NOTE: this binding previously declared no base class, so the type was not a
  // JetScapeTask in Python and JetScape.Add(mgr) could not convert it -- the class was
  // read-only by accident. JetScapeModuleBase is added here so the manager can be placed
  // in a hand-wired pipeline. Existing read-only uses are unaffected.
  py::class_<JetEnergyLossManager, JetScapeModuleBase,
             py::smart_holder>(
      m, "JetEnergyLossManager",
      "Energy-loss manager; one JetEnergyLoss child per shower-initiating parton.")
      .def(py::init<>())
      .def("get_showers",
           [](JetEnergyLossManager &mgr) {
             std::vector<std::shared_ptr<PartonShower>> out;
             for (auto &it : mgr.GetTaskList()) {
               auto jl = std::dynamic_pointer_cast<JetEnergyLoss>(it);
               if (jl && jl->GetShower())
                 out.push_back(jl->GetShower());
             }
             return out;
           },
           "List of per-event PartonShowers (one per shower-initiating parton). "
           "Valid after the manager's Exec, before ClearPerEvent.")
      .def("get_shower_initiating_partons",
           [](JetEnergyLossManager &mgr) {
             std::vector<std::shared_ptr<Parton>> out;
             for (auto &it : mgr.GetTaskList()) {
               auto jl = std::dynamic_pointer_cast<JetEnergyLoss>(it);
               if (jl)
                 out.push_back(jl->GetShowerInitiatingParton());
             }
             return out;
           },
           "The hard (shower-initiating) partons, one per shower.")
      .def("final_partons_numpy",
           [](JetEnergyLossManager &mgr) {
             // Exactly what JetEnergyLossManager::GetFinalStatePartons hands the
             // hadronization: one list per JetEnergyLoss task, in task order,
             // GetShower()->GetFinalPartons() each (an empty shower stays empty).
             std::vector<std::array<double, kFinalPartonNCols>> rows;
             int ishower = 0;
             for (auto &it : mgr.GetTaskList()) {
               auto jl = std::dynamic_pointer_cast<JetEnergyLoss>(it);
               if (!jl)
                 continue;
               auto shower = jl->GetShower();
               if (shower) {
                 for (auto &p : shower->GetFinalPartons()) {
                   rows.push_back({double(ishower), double(p->pid()),
                                   double(p->pstat()), p->e(), p->px(), p->py(),
                                   p->pz(), p->x_in().t(), p->x_in().x(),
                                   p->x_in().y(), p->x_in().z(), p->restmass(),
                                   double(p->color()), double(p->anti_color())});
                 }
               }
               ++ishower;
             }
             py::array_t<double> out(
                 {(py::ssize_t)rows.size(), (py::ssize_t)kFinalPartonNCols});
             auto a = out.mutable_unchecked<2>();
             for (size_t i = 0; i < rows.size(); ++i)
               for (int c = 0; c < kFinalPartonNCols; ++c) a(i, c) = rows[i][c];
             return out;
           },
           "The final-state partons the hadronization would receive, as an (N, 14) "
           "float64 array with columns FINAL_PARTON_COLUMNS (shower index = position "
           "in the manager's task list). Valid after the manager's Exec, before "
           "ClearPerEvent.")
      .def("number_of_showers", [](JetEnergyLossManager &mgr) {
        int n = 0;
        for (auto &it : mgr.GetTaskList())
          if (std::dynamic_pointer_cast<JetEnergyLoss>(it))
            ++n;
        return n;
      });
}
