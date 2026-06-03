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

#include <memory>
#include <vector>

#include "FourVector.h"
#include "JetClass.h"               // Vertex
#include "JetScapeParticles.h"      // Parton (: JetScapeParticleBase)
#include "PartonShower.h"           // PartonShower (: GTL graph)
#include "JetEnergyLoss.h"          // JetEnergyLoss::GetShower()
#include "JetEnergyLossManager.h"

namespace py = pybind11;
using namespace Jetscape;

void bind_jet(py::module_ &m) {
  // ── Parton (an edge in the shower graph) ──────────────────────────────────
  // Momentum from p_in() [GeV], position from x_in() [fm, t in fm/c].
  // NOTE: Parton::t() is the *virtuality*, so the position-time is x_in().t().
  py::class_<Parton, std::shared_ptr<Parton>>(m, "Parton",
      "A parton: pid/pstat, 4-momentum (px,py,pz,e) [GeV], position (x,y,z) [fm], t [fm/c].")
      .def("pid",   [](Parton &p) { return p.pid(); })
      .def("pstat", [](Parton &p) { return p.pstat(); })
      .def("px",    [](Parton &p) { return p.p_in().x(); })
      .def("py",    [](Parton &p) { return p.p_in().y(); })
      .def("pz",    [](Parton &p) { return p.p_in().z(); })
      .def("e",     [](Parton &p) { return p.p_in().t(); })
      .def("x",     [](Parton &p) { return p.x_in().x(); })
      .def("y",     [](Parton &p) { return p.x_in().y(); })
      .def("z",     [](Parton &p) { return p.x_in().z(); })
      .def("t",     [](Parton &p) { return p.x_in().t(); },
           "Position time x_in().t() [fm/c] (NOT virtuality).");

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

  // ── JetEnergyLossManager — per-event access to the finished showers ───────
  py::class_<JetEnergyLossManager, std::shared_ptr<JetEnergyLossManager>>(
      m, "JetEnergyLossManager",
      "Energy-loss manager; one JetEnergyLoss child per shower-initiating parton.")
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
      .def("number_of_showers", [](JetEnergyLossManager &mgr) {
        int n = 0;
        for (auto &it : mgr.GetTaskList())
          if (std::dynamic_pointer_cast<JetEnergyLoss>(it))
            ++n;
        return n;
      });
}
