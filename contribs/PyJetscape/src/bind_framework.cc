/*******************************************************************************
 * bind_framework.cc
 *
 * Binds:
 *   - JetScapeTask      (base of all modules: Add, SetId, GetId)
 *   - JetScapeModuleBase (SetXMLMainFileName, SetXMLUserFileName, Init, Exec)
 *                        WITH PyJetScapeModuleBase trampoline so Python
 *                        subclasses can override Init(), Exec(), Clear(),
 *                        and Finish().
 *   - JetScape (Init, Exec, Finish, SetNumberOfEvents, SetReuseHydro, …)
 *   - create_module(name) — wraps JetScapeModuleFactory::createInstance
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FluidDynamics.h"
#include "InitialState.h"
#include "JetScape.h"
#include "JetScapeModuleBase.h"
#include "JetScapeTask.h"
#include "JetScapeXML.h"
#include "MusicWrapper.h"
#include "PreequilibriumDynamics.h"
#include "TrentoInitial.h"
#include "tinyxml2.h"

namespace py = pybind11;
using namespace Jetscape;

// Walk a sequence of XML tag names from root, returning the deepest element
// found (nullptr if any step fails).  Mirrors the logic in JetScapeXML.cc but
// accepts std::vector<std::string> so it can be called from Python.
static tinyxml2::XMLElement *
xml_walk(tinyxml2::XMLElement *root, const std::vector<std::string> &path) {
  if (!root) return nullptr;
  tinyxml2::XMLElement *cur = nullptr;
  for (const auto &tag : path) {
    cur = (cur ? cur : root)->FirstChildElement(tag.c_str());
    if (!cur) return nullptr;
  }
  return cur;
}

// Lookup path in user XML (first) then main XML.
static tinyxml2::XMLElement *
xml_find(const std::vector<std::string> &path) {
  auto *xml = JetScapeXML::Instance();
  tinyxml2::XMLElement *el = xml_walk(xml->GetXMLRootUser(), path);
  if (!el)              el = xml_walk(xml->GetXMLRootMain(), path);
  return el;
}

static std::string xml_path_str(const std::vector<std::string> &path) {
  std::string s;
  for (const auto &t : path) s += "/" + t;
  return s;
}

// ── PyJetScapeModuleBase trampoline ──────────────────────────────────────────
// Allows Python subclasses to override Init(), Exec(), Clear(), and Finish().
// Used by PyBulkRootWriter and any other Python module added to a JetScape
// pipeline via JetScape.Add().
//
// Example Python subclass:
//
//   class MyModule(pyjetscape.JetScapeModuleBase):
//       def Init(self):
//           self.SetId("MyModule")
//       def Exec(self):
//           sm    = pyjetscape.JetScapeSignalManager.Instance()
//           hydro = sm.GetHydroPointer()
//           ...
//       def Finish(self):
//           ...  # called by JetScape.Finish()
//
class PyJetScapeModuleBase : public JetScapeModuleBase {
public:
  using JetScapeModuleBase::JetScapeModuleBase; // inherit constructors

  void Init() override {
    PYBIND11_OVERRIDE(void, JetScapeModuleBase, Init);
  }
  void Exec() override {
    PYBIND11_OVERRIDE(void, JetScapeModuleBase, Exec);
  }
  void Clear() override {
    PYBIND11_OVERRIDE(void, JetScapeModuleBase, Clear);
  }
  void Finish() override {
    PYBIND11_OVERRIDE(void, JetScapeModuleBase, Finish);
  }
};

void bind_framework(py::module_ &m) {

  // ── JetScapeTask ────────────────────────────────────────────────────────────
  // Base of every JETSCAPE module; provides Add(), SetId(), GetId().
  py::class_<JetScapeTask, std::shared_ptr<JetScapeTask>>(m, "JetScapeTask")
      .def("Add", &JetScapeTask::Add,
           "Add a sub-module/task to this task's list.",
           py::arg("task"))
      .def("SetId", &JetScapeTask::SetId,
           "Set the string identifier for this task.",
           py::arg("id"))
      .def("GetId", &JetScapeTask::GetId,
           "Return the string identifier of this task.")
      .def("GetNumberOfTasks", &JetScapeTask::GetNumberOfTasks,
           "Return the number of sub-tasks currently registered.")
      .def("GetTaskList",
           [](JetScapeTask &t) -> std::vector<std::shared_ptr<JetScapeTask>> {
             return t.GetTaskList();
           },
           "Return the list of sub-tasks as Python objects.")
      .def("GetTaskAt", &JetScapeTask::GetTaskAt,
           "Return the sub-task at index i.",
           py::arg("i"))
      .def("ClearTaskList", &JetScapeTask::ClearTaskList,
           "Remove all sub-tasks.")
      .def("GetActive", &JetScapeTask::GetActive,
           "Return whether this task is active.")
      .def("SetActive", &JetScapeTask::SetActive,
           "Enable or disable this task.",
           py::arg("active"));

  // ── JetScapeModuleBase ──────────────────────────────────────────────────────
  // Adds XML configuration accessors on top of JetScapeTask.
  // Uses PyJetScapeModuleBase as the trampoline so Python subclasses may
  // override Init(), Exec(), Clear(), and Finish().
  py::class_<JetScapeModuleBase, JetScapeTask, PyJetScapeModuleBase,
             std::shared_ptr<JetScapeModuleBase>>(m, "JetScapeModuleBase",
      R"pbdoc(
        Base class for all JETSCAPE physics modules.

        Subclass this in Python to create a custom module that can be inserted
        into a JETSCAPE pipeline via JetScape.Add():

            class MyModule(pyjetscape.JetScapeModuleBase):
                def Init(self):
                    self.SetId("MyModule")
                def Exec(self):
                    sm    = pyjetscape.JetScapeSignalManager.Instance()
                    hydro = sm.GetHydroPointer()
                    ...
                def Finish(self):
                    ...  # called by JetScape.Finish()
      )pbdoc")
      .def(py::init<>())
      .def("SetXMLMainFileName", &JetScapeModuleBase::SetXMLMainFileName,
           "Set the main (default) XML configuration file path.",
           py::arg("filename"))
      .def("GetXMLMainFileName", &JetScapeModuleBase::GetXMLMainFileName,
           "Return the main XML configuration file path.")
      .def("SetXMLUserFileName", &JetScapeModuleBase::SetXMLUserFileName,
           "Set the user (override) XML configuration file path.",
           py::arg("filename"))
      .def("GetXMLUserFileName", &JetScapeModuleBase::GetXMLUserFileName,
           "Return the user XML configuration file path.")
      .def("Init", &JetScapeModuleBase::Init,
           "Initialise this module (reads XML, sets up sub-tasks).")
      .def("Exec", &JetScapeModuleBase::Exec,
           "Execute this module for one event.")
      .def("Clear", &JetScapeModuleBase::Clear,
           "Clear per-event state (called between events).")
      .def("Finish", &JetScapeModuleBase::Finish,
           "Finalise the module (called by JetScape::Finish()).")
      // ── JetScapeXML accessors (user file first, then main file) ────────────
      // Mirrors JetScapeModuleBase::GetXMLElement{Text,Int,Double} but accepts
      // a Python list of tag names instead of a C++ initializer_list.
      // Safe to call from Python overrides of InitializeHydro() / Init() since
      // the XML files are already open by the time those methods run.
      .def("get_xml_element_text",
           [](JetScapeModuleBase &, std::vector<std::string> path,
              bool required) -> std::string {
             auto *el = xml_find(path);
             if (el && el->GetText()) return el->GetText();
             if (required)
               throw std::runtime_error(
                   "XML element " + xml_path_str(path) + " not found");
             return "";
           },
           R"pbdoc(
             Read a text value from the loaded JETSCAPE XML.

             Searches the user XML first, then the main XML (same priority as
             the C++ JetScapeModuleBase::GetXMLElementText helper).

             Parameters
             ----------
             path : list[str]
                 Sequence of tag names, e.g. ["Hydro", "FNO", "model_file"].
             required : bool, default True
                 If True, raises RuntimeError when the element is not found.

             Returns
             -------
             str
                 Element text content, or "" when not found and not required.
           )pbdoc",
           py::arg("path"), py::arg("required") = true)
      .def("get_xml_element_int",
           [](JetScapeModuleBase &, std::vector<std::string> path,
              bool required) -> int {
             auto *el = xml_find(path);
             if (el) { int v = 0; el->QueryIntText(&v); return v; }
             if (required)
               throw std::runtime_error(
                   "XML element " + xml_path_str(path) + " not found");
             return 0;
           },
           R"pbdoc(
             Read an integer value from the loaded JETSCAPE XML.

             Searches the user XML first, then the main XML.

             Parameters
             ----------
             path : list[str]
                 Sequence of tag names, e.g. ["Hydro", "FNO", "nx"].
             required : bool, default True
                 If True, raises RuntimeError when the element is not found.

             Returns
             -------
             int
                 Parsed integer, or 0 when not found and not required.
           )pbdoc",
           py::arg("path"), py::arg("required") = true)
      .def("get_xml_element_double",
           [](JetScapeModuleBase &, std::vector<std::string> path,
              bool required) -> double {
             auto *el = xml_find(path);
             if (el) { double v = 0.; el->QueryDoubleText(&v); return v; }
             if (required)
               throw std::runtime_error(
                   "XML element " + xml_path_str(path) + " not found");
             return 0.;
           },
           R"pbdoc(
             Read a double value from the loaded JETSCAPE XML.

             Searches the user XML first, then the main XML.

             Parameters
             ----------
             path : list[str]
                 Sequence of tag names, e.g. ["Hydro", "FNO", "dtau"].
             required : bool, default True
                 If True, raises RuntimeError when the element is not found.

             Returns
             -------
             float
                 Parsed double, or 0.0 when not found and not required.
           )pbdoc",
           py::arg("path"), py::arg("required") = true);

  // ── JetScape ────────────────────────────────────────────────────────────────
  // Top-level framework controller.  Drive a simulation with:
  //   js = JetScape()
  //   js.SetXMLMainFileName("config/jetscape_main.xml")
  //   js.SetXMLUserFileName("config/jetscape_user.xml")
  //   js.Init(); js.Exec(); js.Finish()
  py::class_<JetScape, JetScapeModuleBase, std::shared_ptr<JetScape>>(
      m, "JetScape")
      .def(py::init<>())
      .def("SetXMLMainFileName", &JetScape::SetXMLMainFileName,
           py::arg("filename"))
      .def("SetXMLUserFileName", &JetScape::SetXMLUserFileName,
           py::arg("filename"))
      .def("Init", &JetScape::Init,
           "Initialise the full task list (reads XML, wires signals).")
      .def("Exec", &JetScape::Exec,
           "Run all events.")
      .def("Finish", &JetScape::Finish,
           "Finalise output and clean up.")
      .def("SetNumberOfEvents", &JetScape::SetNumberOfEvents,
           "Override the number of events from Python.",
           py::arg("n_events"))
      .def("GetNumberOfEvents", &JetScape::GetNumberOfEvents,
           "Return the configured number of events.")
      .def("SetReuseHydro", &JetScape::SetReuseHydro,
           "Enable or disable hydro reuse across jet events.",
           py::arg("reuse"))
      .def("GetReuseHydro", &JetScape::GetReuseHydro,
           "Return whether hydro reuse is enabled.")
      .def("SetNReuseHydro", &JetScape::SetNReuseHydro,
           "Set how many jet events reuse each hydro event.",
           py::arg("n"))
      .def("GetNReuseHydro", &JetScape::GetNReuseHydro,
           "Return the hydro-reuse count.");

  // ── Module factory ──────────────────────────────────────────────────────────
  // Instantiate any C++ module that is registered via RegisterJetScapeModule<T>.
  // Returns shared_ptr<JetScapeModuleBase> (nullptr if name is not registered).
  //
  // Usage (Mode B manual pipeline):
  //   ini   = pyjetscape.create_module("TrentoInitial")
  //   preeq = pyjetscape.create_module("FreestreamMilne")
  //   jmgr  = pyjetscape.create_module("JetEnergyLossManager")
  m.def("create_module",
        [](const std::string &name) -> py::object {
          auto mod = JetScapeModuleFactory::createInstance(name);
          if (!mod)
            throw py::value_error(
                "Module '" + name +
                "' is not registered in JetScapeModuleFactory. "
                "Make sure the corresponding library (e.g. libFnoModule) is "
                "loaded before calling create_module().");
          // Downcast to the most derived registered Python type so that all
          // bound methods are accessible (e.g. FluidDynamics::get_bulk_info,
          // InitialState::get_entropy_density_numpy, etc.).
          //
          // Concrete types are checked FIRST so that pybind11 resolves the
          // Python type as e.g. MpiMusic (not just FluidDynamics), giving
          // access to module-specific methods like set_preserve_bulk_info().
          if (auto music = std::dynamic_pointer_cast<MpiMusic>(mod))
            return py::cast(music);
          if (auto fd = std::dynamic_pointer_cast<FluidDynamics>(mod))
            return py::cast(fd);
          if (auto preeq =
                  std::dynamic_pointer_cast<PreequilibriumDynamics>(mod))
            return py::cast(preeq);
          if (auto trento =
                  std::dynamic_pointer_cast<TrentoInitial>(mod))
            return py::cast(trento);
          if (auto ini = std::dynamic_pointer_cast<InitialState>(mod))
            return py::cast(ini);
          return py::cast(mod);
        },
        R"pbdoc(
          Create a registered JETSCAPE C++ module by name.

          Parameters
          ----------
          name : str
              Registered module name, e.g. "TrentoInitial", "FreestreamMilne",
              "FnoHydro", "JetEnergyLossManager", "Matter", …

          Returns
          -------
          JetScapeModuleBase
              A shared_ptr-managed instance of the requested module.

          Raises
          ------
          ValueError
              If the name is not found in the module registry.
        )pbdoc",
        py::arg("name"));
}
