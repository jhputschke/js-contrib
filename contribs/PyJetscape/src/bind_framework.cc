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
 *   - JetScapePerEvent (ExecInit, ExecPerEvent, ClearPerEvent, SetStartEvent)
 *                       — drive the event loop one event at a time from Python.
 *   - create_module(name) — wraps JetScapeModuleFactory::createInstance
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <random>
#include <utility>

#include "FluidDynamics.h"
#include "InitialState.h"
#include "JetEnergyLoss.h"
#include "JetEnergyLossManager.h"
#include "Hadronization.h"
#include "HadronizationManager.h"
#include "HardProcess.h"
#include "Afterburner.h"
#include "SoftParticlization.h"
#include "JetScape.h"
#include "JetScapePerEvent.h"
#include "JetScapeModuleBase.h"
#include "JetScapeTask.h"
#include "JetScapeXML.h"
#include "JetScapeTaskSupport.h"
#include "JetScapeWriter.h"
#include "MusicWrapper.h"
#include "PreequilibriumDynamics.h"
#include "TrentoInitial.h"
#include "tinyxml2.h"

namespace py = pybind11;
using namespace Jetscape;

// Walk a sequence of XML tag names from root, returning the deepest element
// found (nullptr if any step fails).  Mirrors the logic in JetScapeXML.cc but
// accepts std::vector<std::string> so it can be called from Python.
// A node and everything below it, copied into `doc` (tinyxml2 4 has no DeepClone).
static tinyxml2::XMLNode *xml_deep_clone(const tinyxml2::XMLNode *src,
                                         tinyxml2::XMLDocument *doc) {
  tinyxml2::XMLNode *copy = src->ShallowClone(doc);
  for (const tinyxml2::XMLNode *c = src->FirstChild(); c; c = c->NextSibling())
    copy->InsertEndChild(xml_deep_clone(c, doc));
  return copy;
}

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
class PyJetScapeModuleBase : public JetScapeModuleBase,
                        public py::trampoline_self_life_support {
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
  // The per-task hooks C++ modules override (JetScapeTask's Init/Exec/Clear/Finish
  // call them), and the per-event and per-time-step hooks of the X-SCAPE loop.
  void InitTask() override {
    PYBIND11_OVERRIDE(void, JetScapeModuleBase, InitTask);
  }
  void ExecuteTask() override {
    PYBIND11_OVERRIDE(void, JetScapeModuleBase, ExecuteTask);
  }
  void ClearTask() override {
    PYBIND11_OVERRIDE(void, JetScapeModuleBase, ClearTask);
  }
  void FinishTask() override {
    PYBIND11_OVERRIDE(void, JetScapeModuleBase, FinishTask);
  }
  void InitPerEvent() override {
    PYBIND11_OVERRIDE(void, JetScapeModuleBase, InitPerEvent);
  }
  void FinishPerEvent() override {
    PYBIND11_OVERRIDE(void, JetScapeModuleBase, FinishPerEvent);
  }
  void CalculateTime() override {
    PYBIND11_OVERRIDE(void, JetScapeModuleBase, CalculateTime);
  }
  void ExecTime() override {
    PYBIND11_OVERRIDE(void, JetScapeModuleBase, ExecTime);
  }
};

// ── Python modules in the module factory ────────────────────────────────────
// JetScapeModuleFactory keeps plain function pointers (no captured state), so a
// creator for a Python class can not be made at run time. A fixed pool of
// creators, one per slot, each calling the Python factory stored in its slot,
// stands in for them. The map itself is protected; a subclass reaches it.
namespace {

constexpr int kPyModuleSlots = 256;

struct PyModuleSlot {
  std::string name;
  py::object factory;  // a class or any callable returning a JetScapeModuleBase
  std::string section;   // main-XML section of its default entry ("" = the root)
  std::string defaults;  // the default entry's children, as XML text
};

// Never destroyed: the Python objects must not be released after the
// interpreter has finalized (the factory map outlives it, too).
std::vector<PyModuleSlot> &py_module_slots() {
  static auto *slots = new std::vector<PyModuleSlot>(kPyModuleSlots);
  return *slots;
}

template <int N> std::shared_ptr<JetScapeModuleBase> create_py_module() {
  py::gil_scoped_acquire gil;
  auto &slot = py_module_slots()[N];
  py::object obj = slot.factory();
  auto mod = obj.cast<std::shared_ptr<JetScapeModuleBase>>();
  if (!mod)
    throw std::runtime_error("register_module: the factory of '" + slot.name +
                             "' returned None");
  return mod;
}

using ModuleCreator = std::shared_ptr<JetScapeModuleBase> (*)();

template <int... I>
std::array<ModuleCreator, sizeof...(I)>
make_creators(std::integer_sequence<int, I...>) {
  return {&create_py_module<I>...};
}

const std::array<ModuleCreator, kPyModuleSlots> &py_module_creators() {
  static const auto creators =
      make_creators(std::make_integer_sequence<int, kPyModuleSlots>{});
  return creators;
}

struct FactoryAccess : JetScapeModuleFactory {
  static map_type *map() { return getMap(); }
};

int py_module_slot_of(const std::string &name) {
  auto &slots = py_module_slots();
  for (int k = 0; k < kPyModuleSlots; ++k)
    if (slots[k].factory && slots[k].name == name)
      return k;
  return -1;
}

} // namespace

// JetScape::Init refuses a user XML tag without a default entry in the main XML
// (CompareElementsFromXML: anywhere in the file). The main XML is parsed once
// (JetScapeXML::OpenXMLMainFile is a no-op when open), so opening it here and
// adding an empty element for each registered Python module it lacks lets the
// user XML name them, as <Eloss><CustomModuleX>.
static bool xml_has_element(const tinyxml2::XMLElement *el, const char *name) {
  for (; el; el = el->NextSiblingElement()) {
    if (std::string(el->Name()) == name || xml_has_element(el->FirstChildElement(), name))
      return true;
  }
  return false;
}

static int add_python_module_defaults(const std::string &main_xml) {
  auto *xml = JetScapeXML::Instance();
  if (!main_xml.empty())
    xml->OpenXMLMainFile(main_xml);
  tinyxml2::XMLElement *root = xml->GetXMLRootMain();
  if (!root)
    return 0;
  int n = 0;
  auto *doc = root->GetDocument();
  for (auto &slot : py_module_slots()) {
    if (!slot.factory || xml_has_element(root, slot.name.c_str()))
      continue;
    tinyxml2::XMLElement *parent = root;
    if (!slot.section.empty()) {
      parent = root->FirstChildElement(slot.section.c_str());
      if (!parent)
        parent = root->InsertEndChild(doc->NewElement(slot.section.c_str()))
                     ->ToElement();
    }
    auto *el = parent->InsertEndChild(doc->NewElement(slot.name.c_str()))->ToElement();
    if (!slot.defaults.empty()) {
      tinyxml2::XMLDocument tmp;
      std::string text = "<d>" + slot.defaults + "</d>";
      if (tmp.Parse(text.c_str()) != tinyxml2::XML_SUCCESS)
        throw py::value_error("register_module('" + slot.name +
                              "'): defaults is no valid XML");
      for (auto *c = tmp.FirstChildElement("d")->FirstChild(); c; c = c->NextSibling())
        el->InsertEndChild(xml_deep_clone(c, doc));
    }
    ++n;
  }
  return n;
}

// The most derived bound Python type of a task: a C++ module whose own class is not
// bound (Matter, LBT, ...) shows up as its bound base (JetEnergyLoss, ...), not as
// JetScapeTask. Python modules come back as themselves.
static py::object as_bound_module(std::shared_ptr<JetScapeTask> task) {
  if (!task)
    return py::none();
  if (auto music = std::dynamic_pointer_cast<MpiMusic>(task))
    return py::cast(music);
  if (auto fd = std::dynamic_pointer_cast<FluidDynamics>(task))
    return py::cast(fd);
  if (auto preeq = std::dynamic_pointer_cast<PreequilibriumDynamics>(task))
    return py::cast(preeq);
  if (auto trento = std::dynamic_pointer_cast<TrentoInitial>(task))
    return py::cast(trento);
  if (auto ini = std::dynamic_pointer_cast<InitialState>(task))
    return py::cast(ini);
  if (auto w = std::dynamic_pointer_cast<JetScapeWriter>(task))
    return py::cast(w);
  if (auto sp = std::dynamic_pointer_cast<SoftParticlization>(task))
    return py::cast(sp);
  if (auto ab = std::dynamic_pointer_cast<Afterburner>(task))
    return py::cast(ab);
  if (auto hp = std::dynamic_pointer_cast<HardProcess>(task))
    return py::cast(hp);
  if (auto hm = std::dynamic_pointer_cast<HadronizationManager>(task))
    return py::cast(hm);
  if (auto h = std::dynamic_pointer_cast<Hadronization>(task))
    return py::cast(h);
  if (auto jlm = std::dynamic_pointer_cast<JetEnergyLossManager>(task))
    return py::cast(jlm);
  if (auto jl = std::dynamic_pointer_cast<JetEnergyLoss>(task))
    return py::cast(jl);
  if (auto mod = std::dynamic_pointer_cast<JetScapeModuleBase>(task))
    return py::cast(mod);
  return py::cast(task);
}

void bind_framework(py::module_ &m) {

  // ── std::mt19937: a task's random generator ─────────────────────────────────
  // JetScapeModuleBase::GetMt19937Generator() hands every task its own engine,
  // seeded from <Random><seed> and the task number. A Python module draws from it
  // as a C++ module does, or seeds its own generator from one draw.
  py::class_<std::mt19937, std::shared_ptr<std::mt19937>>(m, "Mt19937",
      "A task's std::mt19937 engine (JetScapeModuleBase.GetMt19937Generator()).")
      .def("__call__", [](std::mt19937 &g) { return static_cast<uint32_t>(g()); },
           "The next 32-bit draw.")
      .def("uniform",
           [](std::mt19937 &g) {
             return std::uniform_real_distribution<double>(0.0, 1.0)(g);
           },
           "A draw in [0, 1), as the C++ modules' ZeroOneDistribution(*gen).")
      .def("discard", [](std::mt19937 &g, unsigned long long n) { g.discard(n); },
           py::arg("n"))
      .def("seed", [](std::mt19937 &g, uint32_t s) { g.seed(s); }, py::arg("seed"));

  m.def("random_seed", &JetScapeTaskSupport::GetRandomSeed,
        "The run's <Random><seed> as JetScapeTaskSupport read it (0 before "
        "JetScape.Init() or load_xml(); a time-based seed when the XML says 0).");

  // ── JetScapeTask ────────────────────────────────────────────────────────────
  // Base of every JETSCAPE module; provides Add(), SetId(), GetId().
  py::class_<JetScapeTask, py::smart_holder>(m, "JetScapeTask")
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
           [](JetScapeTask &t) {
             py::list out;
             for (auto &task : t.GetTaskList())
               out.append(as_bound_module(task));
             return out;
           },
           "Return the list of sub-tasks as Python objects (each as its most derived "
           "bound type).")
      .def("GetTaskAt",
           [](JetScapeTask &t, int i) { return as_bound_module(t.GetTaskAt(i)); },
           "Return the sub-task at index i (as its most derived bound type).",
           py::arg("i"))
      .def("ClearTaskList", &JetScapeTask::ClearTaskList,
           "Remove all sub-tasks.")
      .def("GetActive", &JetScapeTask::GetActive,
           "Return whether this task is active.")
      .def("SetActive", &JetScapeTask::SetActive,
           "Enable or disable this task.",
           py::arg("active"))
      .def("GetMyTaskNumber", &JetScapeTask::GetMyTaskNumber,
           "The task's number (fixes its random generator's seed).");

  // ── JetScapeModuleBase ──────────────────────────────────────────────────────
  // Adds XML configuration accessors on top of JetScapeTask.
  // Uses PyJetScapeModuleBase as the trampoline so Python subclasses may
  // override Init(), Exec(), Clear(), and Finish().
  py::class_<JetScapeModuleBase, JetScapeTask, PyJetScapeModuleBase,
             py::smart_holder>(m, "JetScapeModuleBase",
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
      // The hooks a module overrides; bound so an override can call the base
      // (super().InitTask()) and a pipeline can drive a module by hand.
      .def("InitTask", &JetScapeModuleBase::InitTask,
           "Per-task initialisation (Init() calls it once).")
      .def("ExecuteTask", &JetScapeModuleBase::ExecuteTask,
           "The task's work for one event (Exec() calls it).")
      .def("ClearTask", &JetScapeModuleBase::ClearTask,
           "Per-event clean-up (Clear() calls it).")
      .def("FinishTask", &JetScapeModuleBase::FinishTask,
           "End-of-run work (Finish() calls it).")
      .def("InitPerEvent", &JetScapeModuleBase::InitPerEvent,
           "Start of an event in the time-step loop.")
      .def("FinishPerEvent", &JetScapeModuleBase::FinishPerEvent,
           "End of an event in the time-step loop.")
      .def("CalculateTime", &JetScapeModuleBase::CalculateTime,
           "Time-step loop: before ExecTime().")
      .def("ExecTime", &JetScapeModuleBase::ExecTime,
           "Time-step loop: the work of one time step.")
      .def("GetMt19937Generator", &JetScapeModuleBase::GetMt19937Generator,
           "This task's random engine (made on first use; needs the seed from "
           "JetScape.Init() or load_xml()).")
      .def_static("GetCurrentEvent", &JetScapeModuleBase::GetCurrentEvent,
           "Return the framework-global current event number (advanced by "
           "ClearPerEvent() / per-event execution).")
      .def_static("IncrementCurrentEvent",
           &JetScapeModuleBase::IncrementCurrentEvent,
           "Increment the framework-global current event number.")
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
  py::class_<JetScape, JetScapeModuleBase, py::smart_holder>(
      m, "JetScape")
      .def(py::init<>())
      .def("SetXMLMainFileName", &JetScape::SetXMLMainFileName,
           py::arg("filename"))
      .def("SetXMLUserFileName", &JetScape::SetXMLUserFileName,
           py::arg("filename"))
      .def("Init",
           [](JetScape &js) {
             add_python_module_defaults(js.GetXMLMainFileName());
             js.Init();
           },
           "Initialise the full task list (reads XML, wires signals). Python modules "
           "registered with register_module() get an empty default entry in the main "
           "XML first, so the user XML may name them.")
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

  // ── JetScapePerEvent ──────────────────────────────────────────────────────
  // JetScape driver that exposes the event loop one event at a time, so the
  // loop can be steered from Python and the per-event module results can be
  // read before their memory is released:
  //
  //   js = JetScapePerEvent()
  //   js.SetXMLMainFileName("config/jetscape_main.xml")
  //   js.SetXMLUserFileName("config/jetscape_user.xml")
  //   js.Init()
  //   js.ExecInit()                       # optional; ExecPerEvent lazy-inits
  //   for _ in range(js.GetNumberOfEvents()):
  //       js.ExecPerEvent()               # run one event; data stays in memory
  //       sm = JetScapeSignalManager.Instance()   # ... read module data here ...
  //       js.ClearPerEvent()              # release memory, advance counter
  //   js.Finish()
  //
  // Note: Exec() is disabled on this class (it warns and exits); use the
  // ExecInit/ExecPerEvent/ClearPerEvent API instead.
  py::class_<JetScapePerEvent, JetScape, py::smart_holder>(
      m, "JetScapePerEvent",
      R"pbdoc(
        JetScape driver that runs the event loop one event at a time.

        Unlike JetScape (whose Exec() owns the whole loop and clears each
        event's memory before the next), JetScapePerEvent lets the external
        Python loop run a single event, inspect the live module results, and
        then explicitly release the memory:

            js = JetScapePerEvent()
            js.SetXMLMainFileName(main_xml)
            js.SetXMLUserFileName(user_xml)
            js.Init()
            for _ in range(js.GetNumberOfEvents()):
                js.ExecPerEvent()
                # ... read module data for the current event here ...
                js.ClearPerEvent()
            js.Finish()

        SetStartEvent(n) shifts the (framework-global) event counter so the
        first event runs as event n (default 0). Exec() is disabled on this
        class and will warn-and-exit if called.
      )pbdoc")
      .def(py::init<>())
      .def("ExecInit", &JetScapePerEvent::ExecInit,
           "One-time pre-loop setup (collect writers, CheckExec, snapshot "
           "active flags). Idempotent; ExecPerEvent() also calls it lazily.")
      .def("ExecPerEvent", &JetScapePerEvent::ExecPerEvent,
           "Run a single event, leaving all module data in memory for "
           "inspection. The event number is taken from the global counter.")
      .def("ClearPerEvent", &JetScapePerEvent::ClearPerEvent,
           "Release the per-event module memory and advance the event "
           "counter. Call after reading the data for the current event.")
      .def("SetStartEvent", &JetScapePerEvent::SetStartEvent,
           "Set the starting event number (default 0). Must be called before "
           "the first ExecPerEvent().",
           py::arg("start_event"))
      .def("GetStartEvent", &JetScapePerEvent::GetStartEvent,
           "Return the configured starting event number.");

  // ── Module factory ──────────────────────────────────────────────────────────
  // Instantiate any C++ module that is registered via RegisterJetScapeModule<T>.
  // Returns shared_ptr<JetScapeModuleBase> (nullptr if name is not registered).
  //
  // Usage (Mode B manual pipeline):
  //   ini   = pyjetscape.create_module("TrentoInitial")
  //   preeq = pyjetscape.create_module("FreestreamMilne")
  //   jmgr  = pyjetscape.create_module("JetEnergyLossManager")
  // ── XML singleton loader ──────────────────────────────────────────────────
  // JetScape::Init() normally opens the XML files, but some C++ modules read their
  // parameters in their CONSTRUCTOR -- CausalLiquefier's 0-argument ctor calls
  // InitializeParameters(), which reads <Liquefier><CausalLiquefier> (CausalLiquefier.cc:67).
  // A hand-wired Python pipeline builds its modules before JetScape exists, so without this
  // the liquefier would silently fall back to its defaults after a
  // "XML User file not found/not properly opened!" warning.
  //
  // JetScapeModuleBase::SetXMLMainFileName() does NOT help: those members are per-instance,
  // not the singleton.
  m.def("load_xml",
        [](const std::string &main_xml, const std::string &user_xml, bool init_random) {
          auto *xml = JetScapeXML::Instance();
          xml->SetXMLMainFileName(main_xml);
          xml->OpenXMLMainFile();
          if (!user_xml.empty()) {
            xml->SetXMLUserFileName(user_xml);
            xml->OpenXMLUserFile();
          }
          // JetScape::Init() also seeds the task-support RNG from <Random><seed>. Anything
          // that draws random numbers before then -- InitialState::SampleABinaryCollisionPoint,
          // for one -- throws "Trying to use JetScapeTaskSupport::GetMt19937Generator before
          // initialization". Re-reading the seed later is harmless.
          if (init_random) JetScapeTaskSupport::ReadSeedFromXML();
          return xml->GetXMLRootMain() != nullptr;
        },
        R"pbdoc(
          Open the main (and optionally user) XML into the JetScapeXML singleton now.

          Call this BEFORE constructing any C++ module that reads XML in its constructor --
          CausalLiquefier is the one that matters.  JetScape.Init() does the same thing, but
          that is too late for modules built beforehand.  Calling it twice is harmless.

          With init_random=True (the default) it also seeds JetScapeTaskSupport from
          <Random><seed>, so code that draws random numbers before JetScape.Init() -- such as
          InitialState.sample_binary_collision_point() -- works.

          Returns True if the main file parsed.
        )pbdoc",
        py::arg("main_xml"), py::arg("user_xml") = std::string(),
        py::arg("init_random") = true);

  // ── Modules made outside the physics (run_prod_jet.py --hadrons-only) ──────────
  // A task's ID fixes its random seed (one generator per task), and every task made
  // shifts the IDs of all later ones -- the jet-loss clones made per event among them.
  // Hadronization modules made in the production process take their seeds explicitly,
  // so the driver sets the counter back after making them: the physics then draws the
  // random numbers it would without them.
  m.def("next_task_number",
        []() {
#ifdef JETSCAPETASKSUPPORT_HAS_NEXT_TASK_NUMBER
          return JetScapeTaskSupport::Instance()->GetNextTaskNumber();
#else
          throw std::runtime_error("next_task_number: this X-SCAPE has no "
                                   "JetScapeTaskSupport::GetNextTaskNumber");
          return 0;
#endif
        },
        "The task ID the next task made will get (JetScapeTaskSupport).");
  m.def("set_next_task_number",
        [](int n) {
#ifdef JETSCAPETASKSUPPORT_HAS_NEXT_TASK_NUMBER
          JetScapeTaskSupport::Instance()->SetNextTaskNumber(n);
#else
          throw std::runtime_error("set_next_task_number: this X-SCAPE has no "
                                   "JetScapeTaskSupport::SetNextTaskNumber");
#endif
        },
        "Set the task ID the next task gets: back to next_task_number()'s value before "
        "modules made outside the physics, so later tasks keep their random seeds.",
        py::arg("n"));
#ifdef JETSCAPETASKSUPPORT_HAS_NEXT_TASK_NUMBER
  m.attr("HAS_NEXT_TASK_NUMBER") = true;
#else
  m.attr("HAS_NEXT_TASK_NUMBER") = false;
#endif

  m.def("xml_add_user_blocks",
        [](const std::string &text) {
          auto *xml = JetScapeXML::Instance();
          tinyxml2::XMLElement *root = xml->GetXMLRootUser();
          if (!root)
            throw std::runtime_error("xml_add_user_blocks: no user XML is open (call after "
                                     "JetScape.Init())");
          tinyxml2::XMLDocument doc;
          if (doc.Parse(text.c_str()) != tinyxml2::XML_SUCCESS)
            throw py::value_error("xml_add_user_blocks: the text is no valid XML (tinyxml2 "
                                  "error " + std::to_string(int(doc.ErrorID())) + ")");
          tinyxml2::XMLElement *src = doc.FirstChildElement("jetscape");
          if (!src)
            throw py::value_error("xml_add_user_blocks: the text needs a <jetscape> root");
          int n = 0;
          for (auto *c = src->FirstChildElement(); c; c = c->NextSiblingElement()) {
            if (root->FirstChildElement(c->Name()))
              throw py::value_error(std::string("xml_add_user_blocks: the user XML has <") +
                                    c->Name() + "> already");
          }
          for (auto *c = src->FirstChildElement(); c; c = c->NextSiblingElement()) {
            root->InsertEndChild(xml_deep_clone(c, root->GetDocument()));
            ++n;
          }
          return n;
        },
        R"pbdoc(
          Append the children of ``<jetscape>`` in ``text`` to the open user XML, in memory.

          After JetScape.Init(), so the automatic task list never sees them: modules made
          afterwards (create_module + Init) read their settings from them, while the event
          loop keeps exactly the tasks the file had.  A block the user XML has already is
          refused.  Returns the number of blocks added.
        )pbdoc",
        py::arg("text"));

  // Writers built with create_module() get no file name: in XML mode that is set by
  // JetScape::DetermineWritersFromXML(), which does not run for a hand-wired pipeline.
  m.def("set_writer_output_file",
        [](std::shared_ptr<JetScapeModuleBase> mod, const std::string &file_name) {
          auto writer = std::dynamic_pointer_cast<JetScapeWriter>(mod);
          if (!writer)
            throw py::value_error("set_writer_output_file: module '" + mod->GetId() +
                                  "' is not a JetScapeWriter");
          writer->SetOutputFileName(file_name);
        },
        R"pbdoc(
          Set the output file of a writer created with create_module(), e.g.
          create_module("JetScapeWriterFinalStateHadronsAscii").  Call before
          JetScape.Init(), which opens the file.
        )pbdoc",
        py::arg("writer"), py::arg("file_name"));

  m.def("create_module",
        [](const std::string &name) -> py::object {
          auto mod = JetScapeModuleFactory::createInstance(name);
          if (!mod)
            throw py::value_error(
                "Module '" + name +
                "' is not registered in JetScapeModuleFactory. "
                "Make sure the corresponding library (e.g. libFnoModule) is "
                "loaded before calling create_module().");
          // The most derived bound Python type, so that all bound methods are
          // accessible (MpiMusic's set_preserve_bulk_info(), JetEnergyLoss's
          // DoEnergyLoss() for Matter, ...).
          return as_bound_module(mod);
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

  m.def("register_module",
        [](const std::string &name, py::object factory, const std::string &section,
           const std::string &defaults) {
          if (!PyCallable_Check(factory.ptr()))
            throw py::type_error("register_module: the factory must be a class or a "
                                 "callable returning a JetScapeModuleBase");
          auto *map = FactoryAccess::map();
          int slot = py_module_slot_of(name);
          if (slot < 0 && map->count(name))
            throw py::value_error("register_module: '" + name + "' is a C++ module; "
                                  "choose another name");
          auto &slots = py_module_slots();
          if (slot < 0) {
            for (int k = 0; k < kPyModuleSlots && slot < 0; ++k)
              if (!slots[k].factory)
                slot = k;
            if (slot < 0)
              throw std::runtime_error("register_module: all " +
                                       std::to_string(kPyModuleSlots) +
                                       " slots for Python modules are taken");
          }
          if (!defaults.empty()) {      // refuse bad XML now, not at JetScape.Init()
            tinyxml2::XMLDocument tmp;
            if (tmp.Parse(("<d>" + defaults + "</d>").c_str()) != tinyxml2::XML_SUCCESS)
              throw py::value_error("register_module: defaults is no valid XML");
          }
          slots[slot].name = name;
          slots[slot].factory = factory;
          slots[slot].section = section;
          slots[slot].defaults = defaults;
          (*map)[name] = py_module_creators()[slot];
        },
        R"pbdoc(
          Register a Python module class (or any callable returning a module) in
          JETSCAPE's module factory under ``name``.

          create_module(name) then makes one, and so does the XML's automatic task
          list: every section takes a module whose element name contains
          ``CustomModule`` (e.g. ``<Eloss><CustomModulePyBrick>``), so register
          modules meant for the XML under such a name. C++ owns what it makes; the
          Python object lives as long as C++ holds it.

          JetScape refuses a user-XML tag the main XML lacks, so JetScape.Init()
          adds a default entry for each registered module to the main XML (in
          memory): <name> under ``section`` (e.g. "Eloss"; "" = the root) with the
          children in ``defaults`` (XML text, e.g. "<E>40</E>"). A module reads its
          parameters with get_xml_element_*([section, name, ...]): the user XML's
          value, else this default.

          Registering a name again replaces the Python factory; a C++ module's name
          is refused.
        )pbdoc",
        py::arg("name"), py::arg("factory"), py::arg("section") = std::string(),
        py::arg("defaults") = std::string());

  m.def("unregister_module",
        [](const std::string &name) {
          int slot = py_module_slot_of(name);
          if (slot < 0)
            return false;
          FactoryAccess::map()->erase(name);
          auto &slots = py_module_slots();
          slots[slot].factory = py::object();
          slots[slot].name.clear();
          slots[slot].section.clear();
          slots[slot].defaults.clear();
          return true;
        },
        "Remove a Python module registered with register_module(); False if the "
        "name was not one.",
        py::arg("name"));

  m.def("registered_modules",
        [](bool python_only) {
          std::vector<std::string> names;
          for (auto &kv : *FactoryAccess::map())
            if (!python_only || py_module_slot_of(kv.first) >= 0)
              names.push_back(kv.first);
          return names;
        },
        "The names in JETSCAPE's module factory (C++ modules of this build and the "
        "Python ones registered); python_only=True lists only the latter.",
        py::arg("python_only") = false);
}
