"""
    pyjetscape_core — Python bindings for the JETSCAPE heavy-ion simulation framework.

    Exposes C++ classes and the module factory so that Python code can:
      * Drive a full JETSCAPE simulation (JetScape, Init/Exec/Finish)
      * Instantiate registered C++ modules by name (create_module)
      * Subclass FluidDynamics in Python (PyFNOHydro trampoline)
      * Subclass JetScapeModuleBase in Python (PyBulkRootWriter and similar)
      * Add the C++ FastRootBulkWriter to a pipeline (ROOT builds, HAS_ROOT)
      * Access InitialState and PreequilibriumDynamics data as numpy arrays
      * Read EvolutionHistory and SurfaceCellInfo after hydro finishes
      * Query globally registered modules via JetScapeSignalManager
  
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Iterator
import collections.abc
from jetscape.awkward_views import select
from jetscape.awkward_views import to_awkward
from jetscape.hadron_array import HadronArray
from jetscape.pythonic import ShowerNode
from jetscape.pythonic import make_hadron
from jetscape.pythonic import make_parton
from jetscape.pythonic import make_photon
from jetscape.pythonic import module
from jetscape.pythonic import per_event
import numpy
import numpy.typing
import typing
__all__: list[str] = ['Afterburner', 'CausalLiquefier', 'Droplet', 'ERROR', 'EVOLVING', 'EvolutionHistory', 'FINAL_PARTON_COLUMNS', 'FINISHED', 'FastRootBulkWriter', 'FluidCellInfo', 'FluidDynamics', 'FourVector', 'HAS_GZIP_READER', 'HAS_NEXT_TASK_NUMBER', 'HAS_ROOT', 'Hadron', 'HadronArray', 'Hadronization', 'HadronizationManager', 'HardProcess', 'HydroStatus', 'ICOLORED_HAS_COLOUR_AUDIT', 'ICOLORED_HAS_COLOUR_CLOSURE', 'ICOLORED_HAS_INPUT_RECORD', 'INITIALIZED', 'ISS_DECAYS_SMASH_LIST', 'InitialState', 'IsrJet', 'IsrManager', 'IsrShowerPSG', 'JetEnergyLoss', 'JetEnergyLossManager', 'JetScape', 'JetScapeEventHeader', 'JetScapeModuleBase', 'JetScapeParticleBase', 'JetScapePerEvent', 'JetScapeReaderAscii', 'JetScapeReaderAsciiGZ', 'JetScapeSignalManager', 'JetScapeTask', 'JetScapeWriter', 'LiquefierBase', 'MpiMusic', 'Mt19937', 'NOT_START', 'PYTHIA_GUN_HAS_PARTON_Y_CUT', 'PYTHIA_GUN_HAS_PTHAT_BINS', 'PYTHIA_GUN_HAS_Z_TRIGGER', 'PYTHIA_ISR_GUN_HAS_PARTON_Y_CUT', 'PYTHIA_ISR_GUN_HAS_PTHAT_BINS', 'PYTHIA_VERSION', 'Parameter', 'Parton', 'PartonShower', 'PartonShowerGenerator', 'PartonShowerGeneratorDefault', 'Photon', 'PreequilibriumDynamics', 'SOFT_HAS_PRIMARIES', 'SOFT_HAS_SAMPLE_CELLS', 'SURFACE_CELL_COLUMNS', 'ShowerNode', 'SoftParticlization', 'SurfaceCellInfo', 'TrentoInitial', 'Vertex', 'create_module', 'hadronization_hadrons_numpy', 'hadronization_input_numpy', 'hadronize_input', 'hadronize_partons', 'hard_partons_numpy', 'hard_process_info', 'iMATTER', 'initial_state_info', 'isr_hard_momentum', 'jet_hadronization_colour_audit', 'jet_hadronization_colour_totals', 'jet_hadronization_last_random_seed', 'jet_hadronization_set_colour_closure', 'load_xml', 'make_hadron', 'make_parton', 'make_photon', 'module', 'next_task_number', 'per_event', 'pythia_gun_bins', 'pythia_gun_trigger', 'random_seed', 'register_module', 'registered_modules', 'select', 'set_next_task_number', 'set_writer_output_file', 'soft_hadrons_numpy', 'soft_last_random_seed', 'soft_primaries_numpy', 'soft_sample_cells', 'soft_set_compact_output', 'soft_set_next_random_seed', 'soft_set_number_of_samples', 'to_awkward', 'unregister_module', 'xml_add_user_blocks']
class Mt19937:
    """
    A task's std::mt19937 engine (JetScapeModuleBase.GetMt19937Generator()).
    """
    def __call__(self) -> int:
        """
        The next 32-bit draw.
        """
    def discard(self, n: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def seed(self, seed: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def uniform(self) -> float:
        """
        A draw in [0, 1), as the C++ modules' ZeroOneDistribution(*gen).
        """
class JetScapeTask:
    def Add(self, task: JetScapeTask) -> None:
        """
        Add a sub-module/task to this task's list.
        """
    def ClearTaskList(self) -> None:
        """
        Remove all sub-tasks.
        """
    def GetActive(self) -> bool:
        """
        Return whether this task is active.
        """
    def GetId(self) -> str:
        """
        Return the string identifier of this task.
        """
    def GetMyTaskNumber(self) -> int:
        """
        The task's number (fixes its random generator's seed).
        """
    def GetNumberOfTasks(self) -> int:
        """
        Return the number of sub-tasks currently registered.
        """
    def GetTaskAt(self, i: typing.SupportsInt | typing.SupportsIndex) -> typing.Any:
        """
        Return the sub-task at index i (as its most derived bound type).
        """
    def GetTaskList(self) -> list:
        """
        Return the list of sub-tasks as Python objects (each as its most derived bound type).
        """
    def SetActive(self, active: bool) -> None:
        """
        Enable or disable this task.
        """
    def SetId(self, id: str) -> None:
        """
        Set the string identifier for this task.
        """
class JetScapeModuleBase(JetScapeTask):
    """
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
          
    """
    @staticmethod
    def GetCurrentEvent() -> int:
        """
        Return the framework-global current event number (advanced by ClearPerEvent() / per-event execution).
        """
    @staticmethod
    def IncrementCurrentEvent() -> None:
        """
        Increment the framework-global current event number.
        """
    @property
    def rng(self) -> np.random.Generator:
        """
        A numpy Generator seeded from this task's own engine (GetMt19937Generator), made
            on first use.  Cached with the task number: a Clone() (which copies __dict__) gets a
            new task number and so its own stream.  An instance attribute `rng` overrides it.
            Python modules only: on a C++ module (no __dict__) it raises AttributeError rather
            than draw from that module's engine.
        """
    def CalculateTime(self) -> None:
        """
        Time-step loop: before ExecTime().
        """
    def Clear(self) -> None:
        """
        Clear per-event state (called between events).
        """
    def ClearTask(self) -> None:
        """
        Per-event clean-up (Clear() calls it).
        """
    def Exec(self) -> None:
        """
        Execute this module for one event.
        """
    def ExecTime(self) -> None:
        """
        Time-step loop: the work of one time step.
        """
    def ExecuteTask(self) -> None:
        """
        The task's work for one event (Exec() calls it).
        """
    def Finish(self) -> None:
        """
        Finalise the module (called by JetScape::Finish()).
        """
    def FinishPerEvent(self) -> None:
        """
        End of an event in the time-step loop.
        """
    def FinishTask(self) -> None:
        """
        End-of-run work (Finish() calls it).
        """
    def GetMt19937Generator(self) -> Mt19937:
        """
        This task's random engine (made on first use; needs the seed from JetScape.Init() or load_xml()).
        """
    def GetXMLMainFileName(self) -> str:
        """
        Return the main XML configuration file path.
        """
    def GetXMLUserFileName(self) -> str:
        """
        Return the user XML configuration file path.
        """
    def Init(self) -> None:
        """
        Initialise this module (reads XML, sets up sub-tasks).
        """
    def InitPerEvent(self) -> None:
        """
        Start of an event in the time-step loop.
        """
    def InitTask(self) -> None:
        """
        Per-task initialisation (Init() calls it once).
        """
    def SetXMLMainFileName(self, filename: str) -> None:
        """
        Set the main (default) XML configuration file path.
        """
    def SetXMLUserFileName(self, filename: str) -> None:
        """
        Set the user (override) XML configuration file path.
        """
    def __init__(self) -> None:
        ...
    def get_xml_element_double(self, path: collections.abc.Sequence[str], required: bool = True) -> float:
        """
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
        """
    def get_xml_element_int(self, path: collections.abc.Sequence[str], required: bool = True) -> int:
        """
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
        """
    def get_xml_element_text(self, path: collections.abc.Sequence[str], required: bool = True) -> str:
        """
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
        """
class JetScape(JetScapeModuleBase):
    def Exec(self) -> None:
        """
        Run all events.
        """
    def Finish(self) -> None:
        """
        Finalise output and clean up.
        """
    def GetNReuseHydro(self) -> int:
        """
        Return the hydro-reuse count.
        """
    def GetNumberOfEvents(self) -> int:
        """
        Return the configured number of events.
        """
    def GetReuseHydro(self) -> bool:
        """
        Return whether hydro reuse is enabled.
        """
    def Init(self) -> None:
        """
        Initialise the full task list (reads XML, wires signals). Python modules registered with register_module() get an empty default entry in the main XML first, so the user XML may name them.
        """
    def SetNReuseHydro(self, n: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set how many jet events reuse each hydro event.
        """
    def SetNumberOfEvents(self, n_events: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Override the number of events from Python.
        """
    def SetReuseHydro(self, reuse: bool) -> None:
        """
        Enable or disable hydro reuse across jet events.
        """
    def SetXMLMainFileName(self, filename: str) -> None:
        ...
    def SetXMLUserFileName(self, filename: str) -> None:
        ...
    def __init__(self) -> None:
        ...
class JetScapePerEvent(JetScape):
    """
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
          
    """
    @property
    def energy_loss_manager(self) -> 'JetEnergyLossManager | None':
        """
        The JetEnergyLossManager (task list, else the signal manager), or None.
        """
    @property
    def hadronization_manager(self) -> 'HadronizationManager | None':
        """
        The HadronizationManager of the task list, or None.
        """
    @property
    def hydro(self) -> 'FluidDynamics | None':
        """
        The hydro the signal manager holds (else the first in the task list), or None.
        """
    @property
    def initial_state(self) -> 'InitialState | None':
        """
        The initial state the signal manager holds (else the task list's), or None.
        """
    @property
    def soft_particlization(self) -> 'SoftParticlization | None':
        """
        The (first) SoftParticlization module of the task list (e.g. iSS), or None.
        """
    def ClearPerEvent(self) -> None:
        """
        Release the per-event module memory and advance the event counter. Call after reading the data for the current event.
        """
    def ExecInit(self) -> None:
        """
        One-time pre-loop setup (collect writers, CheckExec, snapshot active flags). Idempotent; ExecPerEvent() also calls it lazily.
        """
    def ExecPerEvent(self) -> None:
        """
        Run a single event, leaving all module data in memory for inspection. The event number is taken from the global counter.
        """
    def GetStartEvent(self) -> int:
        """
        Return the configured starting event number.
        """
    def SetStartEvent(self, start_event: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the starting event number (default 0). Must be called before the first ExecPerEvent().
        """
    def __enter__(self) -> 'JetScapePerEvent':
        """
        Init() (and the number of events given to per_event()).
        """
    def __exit__(self, exc_type, exc, tb) -> bool:
        """
        Finish().
        """
    def __init__(self) -> None:
        ...
    def events(self, n: int | None = None) -> 'Iterator[int]':
        """
        Yield once per event, after ExecPerEvent() and before ClearPerEvent(), so the
            event's results can be read; n: at most n events (default GetNumberOfEvents()).
            Stops cleanly on break (the event is cleared); Finish() is the context manager's.
        """
class FourVector:
    """
    A four-vector. The constructor takes (x, y, z, t) as C++ FourVector(x_in, y_in, z_in, t_in); momenta are (px, py, pz, E).
    """
    __hash__: typing.ClassVar[None] = None  # type: ignore[assignment]
    @classmethod
    def momentum(cls, E: float, px: float, py: float, pz: float) -> 'FourVector':
        """
        FourVector.momentum(E, px, py, pz) (the constructor takes px, py, pz, E).
        """
    @classmethod
    def position(cls, t: float, x: float, y: float, z: float) -> 'FourVector':
        """
        FourVector.position(t, x, y, z) (the constructor takes x, y, z, t).
        """
    def Set(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex, t: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def __add__(self, arg0: FourVector) -> FourVector:
        ...
    def __array__(self, dtype = None, copy = None) -> np.ndarray:
        ...
    def __eq__(self, other) -> bool:
        ...
    def __getitem__(self, i: int) -> float:
        ...
    def __iadd__(self, arg0: FourVector) -> FourVector:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex, t: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, other: FourVector) -> None:
        ...
    def __isub__(self, arg0: FourVector) -> FourVector:
        ...
    def __iter__(self) -> typing.Iterator[float]:
        """
        (t, x, y, z) = (E, px, py, pz), the order of comp(i).
        """
    def __ne__(self, other) -> bool:
        ...
    def __reduce__(self) -> tuple:
        ...
    def __repr__(self) -> str:
        ...
    def __sub__(self, arg0: FourVector) -> FourVector:
        ...
    def boost(self, vx: typing.SupportsFloat | typing.SupportsIndex, vy: typing.SupportsFloat | typing.SupportsIndex, vz: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def clear(self) -> None:
        ...
    def comp(self, i: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Component i: 0 = t, 1 = x, 2 = y, 3 = z.
        """
    def eta(self) -> float:
        ...
    def eta_boost(self, deta: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def minus(self) -> float:
        ...
    def phi(self) -> float:
        ...
    def plus(self) -> float:
        ...
    def rapidity(self) -> float:
        ...
    def rotate_around_x(self, theta: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def rotate_around_y(self, theta: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def rotate_around_z(self, theta: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def t(self) -> float:
        ...
    def to_list(self) -> list[float]:
        """
        [x, y, z, t]
        """
    def x(self) -> float:
        ...
    def y(self) -> float:
        ...
    def y_boost(self, dy: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def z(self) -> float:
        ...
class JetScapeParticleBase:
    """
    Common base of Parton and Hadron: identity, momentum, position.
    """
    @property
    def momentum(self) -> 'FourVector':
        """
        The momentum as a FourVector (a copy); iterates as (E, px, py, pz).
        """
    @property
    def position(self) -> 'FourVector':
        """
        The position as a FourVector (a copy); iterates as (t, x, y, z).
        """
    def GetControlled(self) -> bool:
        ...
    def GetController(self) -> str:
        ...
    def SetController(self, controller: str = '') -> bool:
        ...
    def UnsetController(self) -> None:
        ...
    def e(self) -> float:
        ...
    def eta(self) -> float:
        ...
    def init_jet_v(self) -> None:
        ...
    def jet_v(self) -> FourVector:
        ...
    def nu(self) -> float:
        ...
    def p(self, i: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Momentum component i: 0 = E, 1 = px, 2 = py, 3 = pz.
        """
    def p_in(self) -> FourVector:
        """
        Momentum as FourVector(px, py, pz, E).
        """
    def phi(self) -> float:
        """
        Azimuth in [0, 2 pi) (fastjet's phi).
        """
    def pid(self) -> int:
        ...
    def pl(self) -> float:
        ...
    def plabel(self) -> int:
        ...
    def pstat(self) -> int:
        ...
    def pt(self) -> float:
        ...
    def px(self) -> float:
        ...
    def py(self) -> float:
        ...
    def pz(self) -> float:
        ...
    def rap(self) -> float:
        ...
    def rapidity(self) -> float:
        ...
    @typing.overload
    def reset_momentum(self, px: typing.SupportsFloat | typing.SupportsIndex, py: typing.SupportsFloat | typing.SupportsIndex, pz: typing.SupportsFloat | typing.SupportsIndex, e: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def reset_momentum(self, p: FourVector) -> None:
        ...
    def restmass(self) -> float:
        """
        Rest mass [GeV] (the PDG mass the particle was built with).
        """
    def set_id(self, id: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_jet_v(self, v: FourVector) -> None:
        ...
    def set_label(self, label: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_stat(self, stat: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_x(self, x: FourVector) -> None:
        """
        Set the position from a FourVector(x, y, z, t).
        """
    def t(self) -> float:
        """
        Position time x_in().t() [fm/c]. For a Parton the virtuality is virtuality() (C++ Parton::t()).
        """
    def time(self) -> float:
        """
        Position time x_in().t().
        """
    def x(self) -> float:
        ...
    def x_in(self) -> FourVector:
        """
        Position as FourVector(x, y, z, t).
        """
    def y(self) -> float:
        ...
    def z(self) -> float:
        ...
class Parton(JetScapeParticleBase):
    """
            A parton: Parton(label, id, stat, p, x) with p = FourVector(px, py, pz, E)
            [GeV] and x = FourVector(x, y, z, t) [fm]; or Parton(label, id, stat, pt,
            eta, phi, e, x=None) with x a list (t, x, y, z) as C++ double[4].
    
            x(), y(), z(), t() are the position; the virtuality is virtuality()
            (C++ Parton::t()), set with set_t().
          
    """
    @staticmethod
    def isLepton(pid: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    @staticmethod
    def isNonParton(pid: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    def __copy__(self) -> Parton:
        ...
    @typing.overload
    def __init__(self, label: typing.SupportsInt | typing.SupportsIndex, id: typing.SupportsInt | typing.SupportsIndex, stat: typing.SupportsInt | typing.SupportsIndex, p: FourVector, x: FourVector) -> None:
        ...
    @typing.overload
    def __init__(self, label: typing.SupportsInt | typing.SupportsIndex, id: typing.SupportsInt | typing.SupportsIndex, stat: typing.SupportsInt | typing.SupportsIndex, pt: typing.SupportsFloat | typing.SupportsIndex, eta: typing.SupportsFloat | typing.SupportsIndex, phi: typing.SupportsFloat | typing.SupportsIndex, e: typing.SupportsFloat | typing.SupportsIndex, x: typing.Any = None) -> None:
        ...
    @typing.overload
    def __init__(self, other: Parton) -> None:
        """
        A copy.
        """
    def __reduce__(self) -> tuple:
        ...
    def __repr__(self) -> str:
        ...
    def anti_color(self) -> int:
        """
        Anti-colour tag (0 = none).
        """
    def color(self) -> int:
        """
        Colour tag (0 = none).
        """
    def copy(self) -> Parton:
        """
        A copy.
        """
    def edgeid(self) -> int:
        ...
    def form_time(self) -> float:
        ...
    def isPhoton(self, pid: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    def max_color(self) -> int:
        ...
    def mean_form_time(self) -> float:
        ...
    def min_anti_color(self) -> int:
        ...
    def min_color(self) -> int:
        ...
    def reset_p(self, px: typing.SupportsFloat | typing.SupportsIndex, py: typing.SupportsFloat | typing.SupportsIndex, pz: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the 3-momentum, keeping E.
        """
    def set_anti_color(self, acol: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_color(self, col: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_form_time(self, form_time: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_max_color(self, col: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_mean_form_time(self) -> None:
        ...
    def set_min_anti_color(self, acol: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_min_color(self, col: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_t(self, t: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        C++ Parton::set_t(): set the virtuality (rescales the momentum).
        """
    def virtuality(self) -> float:
        """
        C++ Parton::t(): the virtuality [GeV^2].
        """
class Photon(Parton):
    """
    A photon from a shower: Photon(label, id, stat, p, x) as Parton.
    """
    def __init__(self, label: typing.SupportsInt | typing.SupportsIndex, id: typing.SupportsInt | typing.SupportsIndex, stat: typing.SupportsInt | typing.SupportsIndex, p: FourVector, x: FourVector) -> None:
        ...
    def __repr__(self) -> str:
        ...
class Hadron(JetScapeParticleBase):
    """
            A hadron: Hadron(label, id, stat, p, x[, mass[, charge, baryon_number,
            strangeness[, participant]]]) with p = FourVector(px, py, pz, E) [GeV] and
            x = FourVector(x, y, z, t) [fm]. Without a mass the PDG mass is used.
          
    """
    def __copy__(self) -> Hadron:
        ...
    @typing.overload
    def __init__(self, label: typing.SupportsInt | typing.SupportsIndex, id: typing.SupportsInt | typing.SupportsIndex, stat: typing.SupportsInt | typing.SupportsIndex, p: FourVector, x: FourVector) -> None:
        ...
    @typing.overload
    def __init__(self, label: typing.SupportsInt | typing.SupportsIndex, id: typing.SupportsInt | typing.SupportsIndex, stat: typing.SupportsInt | typing.SupportsIndex, p: FourVector, x: FourVector, mass: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, label: typing.SupportsInt | typing.SupportsIndex, id: typing.SupportsInt | typing.SupportsIndex, stat: typing.SupportsInt | typing.SupportsIndex, p: FourVector, x: FourVector, mass: typing.SupportsFloat | typing.SupportsIndex, charge: typing.SupportsInt | typing.SupportsIndex, baryon_number: typing.SupportsInt | typing.SupportsIndex, strangeness: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, label: typing.SupportsInt | typing.SupportsIndex, id: typing.SupportsInt | typing.SupportsIndex, stat: typing.SupportsInt | typing.SupportsIndex, p: FourVector, x: FourVector, mass: typing.SupportsFloat | typing.SupportsIndex, charge: typing.SupportsInt | typing.SupportsIndex, baryon_number: typing.SupportsInt | typing.SupportsIndex, strangeness: typing.SupportsInt | typing.SupportsIndex, participant: bool) -> None:
        ...
    @typing.overload
    def __init__(self, other: Hadron) -> None:
        """
        A copy.
        """
    def __reduce__(self) -> tuple:
        ...
    def __repr__(self) -> str:
        ...
    def baryon_number(self) -> int:
        ...
    def charge(self) -> int:
        ...
    def copy(self) -> Hadron:
        """
        A copy.
        """
    def decay_width(self) -> float:
        ...
    def has_no_position(self) -> bool:
        ...
    def has_valid_momentum(self) -> bool:
        ...
    def participant(self) -> bool:
        ...
    def set_baryon_number(self, baryon_number: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_charge(self, charge: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_decay_width(self, width: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_participant(self, participant: bool) -> None:
        ...
    def set_strangeness(self, strangeness: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def strangeness(self) -> int:
        ...
class FluidCellInfo:
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def bulk_Pi(self) -> float:
        """
        Bulk viscous pressure [GeV/fm^3].
        """
    @bulk_Pi.setter
    def bulk_Pi(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def energy_density(self) -> float:
        """
        Local energy density [GeV/fm^3].
        """
    @energy_density.setter
    def energy_density(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def entropy_density(self) -> float:
        """
        Local entropy density [1/fm^3].
        """
    @entropy_density.setter
    def entropy_density(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mu_B(self) -> float:
        """
        Net baryon chemical potential [GeV].
        """
    @mu_B.setter
    def mu_B(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mu_C(self) -> float:
        """
        Net charge chemical potential [GeV].
        """
    @mu_C.setter
    def mu_C(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mu_S(self) -> float:
        """
        Net strangeness chemical potential [GeV].
        """
    @mu_S.setter
    def mu_S(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def pressure(self) -> float:
        """
        Thermal pressure [GeV/fm^3].
        """
    @pressure.setter
    def pressure(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def qgp_fraction(self) -> float:
        """
        QGP fraction (QGP+HRG phase).
        """
    @qgp_fraction.setter
    def qgp_fraction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def temperature(self) -> float:
        """
        Local temperature [GeV].
        """
    @temperature.setter
    def temperature(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def vx(self) -> float:
        """
        Flow velocity vx.
        """
    @vx.setter
    def vx(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def vy(self) -> float:
        """
        Flow velocity vy.
        """
    @vy.setter
    def vy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def vz(self) -> float:
        """
        Flow velocity vz.
        """
    @vz.setter
    def vz(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class SurfaceCellInfo:
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def bulk_Pi(self) -> float:
        """
        Bulk viscous pressure [GeV/fm^3].
        """
    @bulk_Pi.setter
    def bulk_Pi(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def d3sigma_mu(self) -> list[float]:
        """
        Surface normal vector d^3 sigma_mu (4-vector).
        """
    @d3sigma_mu.setter
    def d3sigma_mu(self, arg1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def energy_density(self) -> float:
        """
        Energy density on the surface [GeV/fm^3].
        """
    @energy_density.setter
    def energy_density(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def entropy_density(self) -> float:
        """
        Entropy density on the surface [1/fm^3].
        """
    @entropy_density.setter
    def entropy_density(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def eta(self) -> float:
        """
        Space-time rapidity.
        """
    @eta.setter
    def eta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mu_B(self) -> float:
        ...
    @mu_B.setter
    def mu_B(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mu_Q(self) -> float:
        ...
    @mu_Q.setter
    def mu_Q(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mu_S(self) -> float:
        ...
    @mu_S.setter
    def mu_S(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def pi(self) -> list[float]:
        """
        Shear stress tensor pi (10 independent components).
        """
    @pi.setter
    def pi(self, arg1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def pressure(self) -> float:
        """
        Pressure on the surface [GeV/fm^3].
        """
    @pressure.setter
    def pressure(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def qgp_fraction(self) -> float:
        ...
    @qgp_fraction.setter
    def qgp_fraction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def tau(self) -> float:
        """
        Proper time tau [fm/c].
        """
    @tau.setter
    def tau(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def temperature(self) -> float:
        """
        Temperature on the surface [GeV].
        """
    @temperature.setter
    def temperature(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def umu(self) -> list[float]:
        """
        4-velocity u^mu.
        """
    @umu.setter
    def umu(self, arg1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def x(self) -> float:
        """
        Transverse x [fm].
        """
    @x.setter
    def x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def y(self) -> float:
        """
        Transverse y [fm].
        """
    @y.setter
    def y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class EvolutionHistory:
    """
            Stores the hydrodynamic evolution history on a (tau,x,y,eta) grid.
    
            Two storage modes:
              * ``data``        — vector of FluidCellInfo objects (used by jet quenching)
              * ``data_vector`` — flat float32 vector (used by external hydro modules)
    
            Grid parameters: tau_min, dtau, ntau, x_min, dx, nx, y_min, dy, ny,
                             eta_min, deta, neta, boost_invariant, tau_eta_is_tz.
          
    """
    boost_invariant: bool
    tau_eta_is_tz: bool
    def EtaMax(self) -> float:
        ...
    def EtaMin(self) -> float:
        ...
    def FromVector(self, data_: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], data_info_: collections.abc.Sequence[str], tau_min: typing.SupportsFloat | typing.SupportsIndex, dtau: typing.SupportsFloat | typing.SupportsIndex, x_min: typing.SupportsFloat | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, nx: typing.SupportsInt | typing.SupportsIndex, y_min: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, ny: typing.SupportsInt | typing.SupportsIndex, eta_min: typing.SupportsFloat | typing.SupportsIndex, deta: typing.SupportsFloat | typing.SupportsIndex, neta: typing.SupportsInt | typing.SupportsIndex, tau_eta_is_tz: bool) -> None:
        """
                     Populate data_vector and grid metadata from a flat float32 vector.
        
                     Parameters match EvolutionHistory::FromVector exactly.
                     data_ must have length ntau * nx * ny * neta * len(data_info_).
        """
    def Tau0(self) -> float:
        ...
    def TauMax(self) -> float:
        ...
    def XMax(self) -> float:
        ...
    def XMin(self) -> float:
        ...
    def YMax(self) -> float:
        ...
    def YMin(self) -> float:
        ...
    def __init__(self) -> None:
        ...
    def cell_index(self, id_tau: typing.SupportsInt | typing.SupportsIndex, id_x: typing.SupportsInt | typing.SupportsIndex, id_y: typing.SupportsInt | typing.SupportsIndex, id_eta: typing.SupportsInt | typing.SupportsIndex) -> int:
        """
        Return flat index for grid position (id_tau, id_x, id_y, id_eta).
        """
    def clear_up_evolution_data(self) -> None:
        """
        Clear the stored cells (data and data_vector).
        """
    def data_vector_numpy(self) -> numpy.typing.NDArray[numpy.float32]:
        """
                     Zero-copy numpy float32 view of the raw data_vector buffer.
                     Shape: (ntau * nx * ny * neta * n_fields,).
                     The field layout is described by data_info.
                     Valid as long as the EvolutionHistory object is alive.
        """
    def frame_numpy(self, id_tau: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float32]:
        """
                     One tau frame of bulk_info.data, shape (nx, ny, neta, 4), float32:
                     [energy_density, vx, vy, vz].  The same values as
                     to_numpy_full(5)[id_tau][..., [0, 2, 3, 4]].
        """
    def get(self, tau: typing.SupportsFloat | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, eta: typing.SupportsFloat | typing.SupportsIndex) -> FluidCellInfo:
        """
                     Return interpolated FluidCellInfo at continuous space-time
                     coordinates (tau, x, y, eta).
        
                     Uses trilinear interpolation over the stored grid.  Coordinates
                     outside the grid are clamped to the boundary.
        
                     Parameters
                     ----------
                     tau  : float  — proper time [fm/c]
                     x    : float  — transverse coordinate x [fm]
                     y    : float  — transverse coordinate y [fm]
                     eta  : float  — space-time rapidity (use 0 for boost-invariant)
        
                     Returns
                     -------
                     FluidCellInfo with interpolated energy_density, temperature,
                     vx, vy, vz, etc.
        """
    def get_at_time_step(self, id_tau: typing.SupportsInt | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, eta: typing.SupportsFloat | typing.SupportsIndex) -> FluidCellInfo:
        """
                     Return FluidCellInfo at a fixed tau step and continuous (x, y, eta).
        
                     Parameters
                     ----------
                     id_tau : int   — tau grid index
                     x      : float — transverse x [fm]
                     y      : float — transverse y [fm]
                     eta    : float — space-time rapidity
        """
    def get_data_size(self) -> int:
        """
        Number of stored cells, in either layout.
        """
    def get_fluid_cell(self, id_tau: typing.SupportsInt | typing.SupportsIndex, id_x: typing.SupportsInt | typing.SupportsIndex, id_y: typing.SupportsInt | typing.SupportsIndex, id_eta: typing.SupportsInt | typing.SupportsIndex) -> FluidCellInfo:
        """
        Return FluidCellInfo at grid index (id_tau, id_x, id_y, id_eta).
        """
    def to_numpy(self, n_features: typing.SupportsInt | typing.SupportsIndex = 4) -> numpy.typing.NDArray[numpy.float32]:
        """
                     Convert bulk_info.data to a numpy float32 array in a single C++
                     pass, avoiding per-cell pybind11 overhead.
        
                     Parameters
                     ----------
                     n_features : int (1–6, default 4)
                         Number of fields to extract per cell.
                         Layout: [energy_density, temperature, vx, vy,
                                  entropy_density, pressure]
        
                     Returns
                     -------
                     np.ndarray, shape (ntau, nx, ny, n_features), dtype float32
        """
    def to_numpy_full(self, n_features: typing.SupportsInt | typing.SupportsIndex = 5) -> numpy.typing.NDArray[numpy.float32]:
        """
                     Convert bulk_info.data to a numpy float32 array keeping the full
                     eta (rapidity) axis, in a single C++ pass.
        
                     Parameters
                     ----------
                     n_features : int (1–6, default 5)
                         Number of fields per cell.
                         Layout: [energy_density, temperature, vx, vy, vz,
                                  entropy_density]
        
                     Returns
                     -------
                     np.ndarray, shape (ntau, nx, ny, neta, n_features), dtype float32
        """
    def uses_data_vector(self) -> bool:
        """
        True if cells are the flat data_vector described by data_info (e.g. MUSIC's slim copy), False for the FluidCellInfo vector.
        """
    @property
    def data_info(self) -> list[str]:
        """
        Entry names of one data_vector record (empty: the FluidCellInfo layout). Set through SetDataInfo, which also resolves them.
        """
    @data_info.setter
    def data_info(self, arg1: collections.abc.Sequence[str]) -> None:
        ...
    @property
    def deta(self) -> float:
        ...
    @deta.setter
    def deta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def dtau(self) -> float:
        ...
    @dtau.setter
    def dtau(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def dx(self) -> float:
        ...
    @dx.setter
    def dx(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def dy(self) -> float:
        ...
    @dy.setter
    def dy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def eta_min(self) -> float:
        ...
    @eta_min.setter
    def eta_min(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def neta(self) -> int:
        ...
    @neta.setter
    def neta(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def ntau(self) -> int:
        ...
    @ntau.setter
    def ntau(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nx(self) -> int:
        ...
    @nx.setter
    def nx(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def ny(self) -> int:
        ...
    @ny.setter
    def ny(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def tau_min(self) -> float:
        ...
    @tau_min.setter
    def tau_min(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def x_min(self) -> float:
        ...
    @x_min.setter
    def x_min(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def y_min(self) -> float:
        ...
    @y_min.setter
    def y_min(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class InitialState(JetScapeModuleBase):
    """
            Interface for initial-state physics modules (e.g. TrentoInitial).
    
            Provides the entropy/binary-collision density grids that seed
            pre-equilibrium and hydro evolution.
    
            Python subclasses can override Init(), Exec(), and Clear() and
            populate the IC grid with set_entropy_density_from_numpy() instead
            of running an external binary.
          
    """
    @property
    def info(self) -> dict:
        """
        initial_state_info(): b, Npart, Ncoll, centrality, ... of this event.
        """
    def GetEntropyDensityDistribution(self) -> list[float]:
        """
                     Return the entropy density distribution as a flat list.
                     Storage order: for z { for y { for x } }.
                     Length = GetXSize() * GetYSize() * GetZSize().
        """
    def GetEventCentrality(self) -> float:
        """
        Event centrality class (-1 if not available).
        """
    def GetEventId(self) -> int:
        """
        Current event ID.
        """
    def GetNcoll(self) -> float:
        """
        Number of binary collisions (-1 if not available).
        """
    def GetNpart(self) -> float:
        """
        Number of participant nucleons (-1 if not available).
        """
    def GetNumOfBinaryCollisions(self) -> list[float]:
        """
                     Return the un-normalised T_A*T_B binary-collision density as a flat list.
                     Same storage order as GetEntropyDensityDistribution().
        """
    def GetTotalEntropy(self) -> float:
        """
        Total entropy of the event (-1 if not available).
        """
    def GetXMax(self) -> float:
        """
        Maximum |x| of the nuclear profile grid [fm].
        """
    def GetXSize(self) -> int:
        """
        Number of grid points in x: ceil(2*xmax/dx).
        """
    def GetXStep(self) -> float:
        """
        Step size dx [fm].
        """
    def GetYMax(self) -> float:
        """
        Maximum |y| of the nuclear profile grid [fm].
        """
    def GetYSize(self) -> int:
        """
        Number of grid points in y: ceil(2*ymax/dy).
        """
    def GetYStep(self) -> float:
        """
        Step size dy [fm].
        """
    def GetZMax(self) -> float:
        """
        Maximum |z or eta| of the nuclear profile grid.
        """
    def GetZSize(self) -> int:
        """
        Number of grid points in z (1 for boost-invariant).
        """
    def GetZStep(self) -> float:
        """
        Step size dz (or d-eta) in z direction.
        """
    def SampleABinaryCollisionPoint(self) -> tuple[float, float, float, float]:
        """
        The C++ name of sample_binary_collision_point(): (t, x, y, z) [fm]. A Python initial state overrides it to return its own hard vertices.
        """
    def SetEventId(self, event_id: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def SetRanges(self, xmax: typing.SupportsFloat | typing.SupportsIndex, ymax: typing.SupportsFloat | typing.SupportsIndex, zmax: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set grid extents (xmax, ymax, zmax).
        """
    def SetSteps(self, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dz: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set grid step sizes (dx, dy, dz).
        """
    def __init__(self) -> None:
        ...
    def get_entropy_density_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
                     Return the entropy density as a 2D numpy array of shape (ny, nx).
                     Only valid for the boost-invariant (2D) case (GetZSize() == 1).
                     Equivalent to:
                       dist = GetEntropyDensityDistribution()
                       arr  = np.array(dist).reshape(ny, nx)
        """
    def get_entropy_density_numpy_3d(self) -> numpy.typing.NDArray[numpy.float64]:
        """
                     The initial-state grid as a 3D numpy array of shape (nx, ny, neta).
        
                     Storage order is the C-order (nx, ny, neta) ravel,
                         idx = (ny*neta)*ix + neta*iy + ieta
                     which is what set_entropy_density_from_numpy() copies in verbatim and what
                     MUSIC's Initial_profile 42 reads.
        
                     NOTE: get_entropy_density_numpy() (the 2D convenience view) instead assumes
                     dist[iy*nx + ix] and is documented as valid only for GetZSize() == 1.  For a
                     square boost-invariant grid the two differ by a transpose, so do not mix them.
        """
    def sample_binary_collision_point(self) -> tuple[float, float, float, float]:
        """
                     Draw one hard-scattering vertex, exactly as the hard process does.
        
                     Returns (t, x, y, z) in fm.  With no density set this returns the origin (and the
                     framework prints a warning), which is the whole reason
                     set_num_of_binary_collisions_from_numpy() exists.
        
                     Coordinates come from CoordFromIdx, i.e. MUSIC's ``-grid_max + i*step`` axis.
        """
    def set_entropy_density_from_numpy(self, arr: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
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
        """
    def set_num_of_binary_collisions_from_numpy(self, arr: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
                     Set the Ta*Tb binary-collision density that hard processes sample vertices from.
        
                     Same C-order (nx, ny, neta) ravel as set_entropy_density_from_numpy().  Without
                     it, InitialState::SampleABinaryCollisionPoint() warns and puts every jet at the
                     origin, so all showers start at the fireball centre.
        
                     NOTE the half-cell offset: CoordFromIdx() maps index ix to
                     ``-grid_max_x + ix*grid_step_x`` (InitialState.cc:75-77), i.e. MUSIC's
                     ``-n*d/2 + i*d`` axis, while the energy-density grid this contribution uses is
                     cell-centred at ``-(n-1)*d/2 + i*d``.  Sampled vertices therefore sit half a cell
                     from the matching energy-density node.
        """
class Parameter:
    hydro_input_filename: str
    def __init__(self) -> None:
        ...
class HydroStatus:
    """
    Members:
    
      NOT_START
    
      INITIALIZED
    
      EVOLVING
    
      FINISHED
    
      ERROR
    """
    ERROR: typing.ClassVar[HydroStatus]
    EVOLVING: typing.ClassVar[HydroStatus]
    FINISHED: typing.ClassVar[HydroStatus]
    INITIALIZED: typing.ClassVar[HydroStatus]
    NOT_START: typing.ClassVar[HydroStatus]
    __members__: typing.ClassVar[dict[str, HydroStatus]]
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PreequilibriumDynamics(JetScapeModuleBase):
    """
            Interface for pre-equilibrium dynamics modules.
            All stress-energy-tensor fields (e_, P_, ux_, uy_, ueta_, pi??_, bulk_Pi_)
            are public std::vector<double> members — directly accessible as zero-copy
            numpy arrays via the get_*_numpy() methods.
          
    """
    def EvolvePreequilibrium(self) -> None:
        """
        Every event, from ExecuteTask().
        """
    def GetPreequilibriumEndTime(self) -> float:
        """
        Proper time tau at the end of pre-equilibrium [fm/c].
        """
    def GetPreequilibriumEvodtau(self) -> float:
        ...
    def GetPreequilibriumStartTime(self) -> float:
        """
        Proper time tau_0 at the start of pre-equilibrium [fm/c].
        """
    def GetPreequilibriumStatus(self) -> int:
        """
        0 NOT_STARTED, 1 INIT, 2 DONE, 3 ERR.
        """
    def InitializePreequilibrium(self) -> None:
        """
        Once, from Init() (after tau0 and taus are read).
        """
    def SetPreequilibriumStatus(self, status: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __init__(self) -> None:
        ...
    def clear_evolution_data(self) -> None:
        ...
    def get_P_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Pressure [GeV/fm^3], shape (nx*ny,).
        """
    def get_bulk_Pi_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Bulk pressure Pi [GeV/fm^3], shape (nx*ny,).
        """
    def get_e_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Energy density [GeV/fm^3], shape (nx*ny,).
        """
    def get_fluid_cell_with_index(self, idx: typing.SupportsInt | typing.SupportsIndex) -> typing.Any:
        ...
    def get_ini_pointer(self) -> InitialState:
        """
        The initial state (set by Init()).
        """
    def get_ntau(self) -> int:
        ...
    def get_number_of_fluid_cells(self) -> int:
        ...
    def get_pi00_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        pi^{00} [GeV/fm^3], shape (nx*ny,).
        """
    def get_pi01_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        pi^{01}, shape (nx*ny,).
        """
    def get_pi02_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        pi^{02}, shape (nx*ny,).
        """
    def get_pi03_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        pi^{03}, shape (nx*ny,).
        """
    def get_pi11_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        pi^{11}, shape (nx*ny,).
        """
    def get_pi12_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        pi^{12}, shape (nx*ny,).
        """
    def get_pi13_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        pi^{13}, shape (nx*ny,).
        """
    def get_pi22_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        pi^{22}, shape (nx*ny,).
        """
    def get_pi23_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        pi^{23}, shape (nx*ny,).
        """
    def get_pi33_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        pi^{33}, shape (nx*ny,).
        """
    def get_ueta_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        u^eta, shape (nx*ny,).
        """
    def get_utau_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        u^tau, shape (nx*ny,).
        """
    def get_ux_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        u^x, shape (nx*ny,).
        """
    def get_uy_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        u^y, shape (nx*ny,).
        """
    def set_P_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_bulk_Pi_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_e_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_pi00_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_pi01_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_pi02_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_pi03_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_pi11_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_pi12_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_pi13_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_pi22_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_pi23_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_pi33_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_ueta_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_utau_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_ux_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    def set_uy_numpy(self, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Set the field (a copy, flattened).
        """
    @property
    def preequilibrium_tau_0_(self) -> float:
        ...
    @preequilibrium_tau_0_.setter
    def preequilibrium_tau_0_(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def preequilibrium_tau_max_(self) -> float:
        ...
    @preequilibrium_tau_max_.setter
    def preequilibrium_tau_max_(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class FluidDynamics(JetScapeModuleBase):
    """
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
          
    """
    def Clear(self) -> None:
        ...
    def EvolveHydro(self) -> None:
        """
        Override in Python to run hydro evolution (called from Exec()).
        """
    def Exec(self) -> None:
        """
        Call EvolveHydro().
        """
    def FindAConstantTemperatureSurface(self, T_sw: typing.SupportsFloat | typing.SupportsIndex) -> list[SurfaceCellInfo]:
        """
        Find the constant-temperature surface and return list of cells.
        """
    def GetHydroEndTime(self) -> float:
        ...
    def GetHydroFreezeOutTemperature(self) -> float:
        ...
    def GetHydroStartTime(self) -> float:
        ...
    def GetHydroStatus(self) -> int:
        ...
    def Init(self) -> None:
        """
        Read XML, retrieve ini/pre_eq pointers, call InitializeHydro().
        """
    def InitializeHydro(self, parameter_list: Parameter) -> None:
        """
        Override in Python to initialise hydro (called from Init()).
        """
    def SetHydroStartTime(self, tau0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set hydro_tau_0 (uninitialised by the FluidDynamics constructor).
        """
    def SetId(self, id: str) -> None:
        """
        Set the module identifier string (e.g. 'PyFNOHydro').
        """
    def __init__(self) -> None:
        ...
    def add_a_liquefier(self, liquefier: typing.Any) -> None:
        """
        Attach a LiquefierBase (or CausalLiquefier) to this hydro. The hydro keeps a weak reference: keep the liquefier object alive (as JetScape does).
        """
    def clearSurfaceCellVector(self) -> None:
        """
        Clear the freeze-out surface cell list.
        """
    def clear_up_evolution_data(self) -> None:
        """
        Clear the FluidCellInfo data in bulk_info.
        """
    def find_freezeout_surface(self, T_sw: typing.SupportsFloat | typing.SupportsIndex, dtau: typing.SupportsFloat | typing.SupportsIndex = 0.0, dx: typing.SupportsFloat | typing.SupportsIndex = 0.0, deta: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        """
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
        """
    def getSurfaceCellVector(self) -> list[SurfaceCellInfo]:
        """
        Return a copy of the freeze-out surface cell vector.
        """
    def get_bulk_info(self) -> EvolutionHistory:
        """
        Return a reference to the EvolutionHistory (read-only).
        """
    def get_bulk_info_mutable(self) -> EvolutionHistory:
        """
        Return a mutable reference to the EvolutionHistory.
        """
    def get_hydro_cell(self, t: typing.SupportsFloat | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> FluidCellInfo:
        """
        GetHydroInfo() at Cartesian (t, x, y, z), returned by value. Never raises and never returns null: out-of-range gives a zero cell.
        """
    def get_hydro_status_int(self) -> int:
        """
        Return the raw HydroStatus integer.
        """
    def get_ini_pointer(self) -> InitialState:
        """
        Return shared_ptr to the InitialState module.
        """
    def get_liquefier(self) -> typing.Any:
        """
        The attached liquefier, or None.
        """
    def get_out_of_range_count(self) -> int:
        """
        Number of GetHydroInfo() queries that fell outside the stored grid (and so saw vacuum). CheckInRange never throws, so this is the only signal.
        """
    def get_preeq_pointer(self) -> PreequilibriumDynamics:
        """
        Return shared_ptr to the PreequilibriumDynamics module.
        """
    def get_preserve_bulk_info(self) -> bool:
        ...
    def get_surface_cells(self) -> list[SurfaceCellInfo]:
        """
        Return a mutable reference to the surface cell vector.
        """
    def reset_out_of_range_count(self) -> None:
        ...
    def set_hydro_grid_info(self, tau_min: typing.SupportsFloat | typing.SupportsIndex, dtau: typing.SupportsFloat | typing.SupportsIndex, ntau: typing.SupportsInt | typing.SupportsIndex, x_min: typing.SupportsFloat | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, nx: typing.SupportsInt | typing.SupportsIndex, y_min: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, ny: typing.SupportsInt | typing.SupportsIndex, eta_min: typing.SupportsFloat | typing.SupportsIndex, deta: typing.SupportsFloat | typing.SupportsIndex, neta: typing.SupportsInt | typing.SupportsIndex, boost_inv: bool = True, tau_eta_is_tz: bool = False) -> None:
        """
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
        """
    def set_hydro_status_finished(self) -> None:
        """
        Mark hydro as FINISHED so that GetHydroInfo() works.
        """
    def set_hydro_status_initialized(self) -> None:
        """
        Mark hydro as INITIALIZED.
        """
    def set_preserve_bulk_info(self, preserve: bool) -> None:
        """
        When True, Clear() will not wipe bulk_info.data. Lets Python read bulk_info after js.Exec() returns.
        """
    def store_fluid_cells_aos_3d(self, arr: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], names: collections.abc.Sequence[str]) -> None:
        """
        Same input as store_fluid_cells_from_numpy_3d(), but fills bulk_info.data (array of FluidCellInfo) instead of data_vector.
        """
    def store_fluid_cells_from_numpy(self, arr: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        """
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
        """
    def store_fluid_cells_from_numpy_3d(self, arr: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], names: collections.abc.Sequence[str]) -> None:
        """
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
        """
    def store_surface_from_numpy(self, cells: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        """
        Append cells from an (N, 32) array (columns SURFACE_CELL_COLUMNS) to the surface vector, e.g. to replay a stored surface into iSS.
        """
    def surface_to_numpy(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        The freeze-out surface as an (N, 32) float32 array, one row per cell, columns SURFACE_CELL_COLUMNS: exactly the fields iSS reads (iSpectraSamplerWrapper::getSurfCellVector), in its order. SurfaceCellInfo is float, so this copy is lossless. Valid after the hydro's Exec, before ClearPerEvent.
        """
class Droplet:
    """
    A unit of four-momentum handed from the jet to the medium.
    Position is Milne (tau, x, y, eta); momentum is Cartesian (E, px, py, pz).
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, xmu: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(4)"], pmu: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(4)"]) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_pmu(self) -> typing.Annotated[list[float], "FixedSize(4)"]:
        """
        (E, px, py, pz)
        """
    def get_xmu(self) -> typing.Annotated[list[float], "FixedSize(4)"]:
        """
        (tau, x, y, eta)
        """
class LiquefierBase:
    """
    Turns what the jet loses to the medium (droplets) into a hydro source. A Python liquefier subclasses it and overrides smearing_kernel(tau, x, y, eta, droplet) -> [j^tau, j^x, j^y, j^eta] (and the deposition hooks as needed).
    """
    COLUMNS: typing.ClassVar[tuple] = ('tau', 'x', 'y', 'eta', 'E', 'px', 'py', 'pz')
    """
    Column order of droplets_numpy(): (tau, x, y, eta, E, px, py, pz).
    """
    def ClearTask(self) -> None:
        """
        Empty the droplet list. FluidDynamics::ClearTask() is normally what calls this; a Python hydro that overrides Clear() must call it itself, or droplets accumulate across events.
        """
    def __init__(self) -> None:
        ...
    def add_a_droplet(self, droplet: Droplet) -> None:
        ...
    def add_droplets_numpy(self, arr: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> int:
        """
        Append droplets from an (M, 8) array in the droplets_numpy() column order. Used by the replay path.
        """
    def deposit_half_width(self) -> float:
        ...
    def deposit_time(self, droplet: Droplet) -> float:
        ...
    def droplet_fluxes_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Per droplet (droplets_numpy() order): the sum of its point-sampled kernel on the hydro grid, i.e. the fraction of its four-momentum the hydro would receive without normalization; -1 if it was never normalized (no grid, switched off, or it never reached a hydro step).  With normalization on, the hydro receives exactly the droplet (flux > 0, or flux 0: deposited into one cell).
        """
    def droplet_may_contribute(self, droplet: Droplet, tau_lo: typing.SupportsFloat | typing.SupportsIndex, tau_hi: typing.SupportsFloat | typing.SupportsIndex) -> bool:
        ...
    def droplets_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
                     The droplet list as an (M, 8) float64 array.
        
                     Columns are (tau, x, y, eta, E, px, py, pz) -- exactly
                     fast_data.liquefier.droplets.COLUMNS, so the result is a DropletArray payload
                     with no conversion.
        
                     float64 is emitted even though Jetscape::real is float: the values have already
                     been rounded to float, and keeping them wide avoids a second rounding.
        """
    def get_GetHydroCellSignalConnected(self) -> bool:
        """
        True once the framework has wired this liquefier to a hydro. filter_partons() needs that signal to boost partons into the fluid rest frame.
        """
    def get_a_droplet(self, idx: typing.SupportsInt | typing.SupportsIndex) -> Droplet:
        ...
    def get_drop_stat(self) -> int:
        """
        pstat of a parton absorbed into the fluid (-11).
        """
    def get_dropletlist_size(self) -> int:
        ...
    def get_dropletlist_total_energy(self) -> float:
        ...
    def get_miss_stat(self) -> int:
        """
        pstat of the energy-momentum-conservation filler parton (-13).
        """
    def get_neg_stat(self) -> int:
        """
        pstat of a negative / back-reaction parton (-17).
        """
    def get_normalize_on_hydro_grid(self) -> bool:
        """
        Whether each droplet is normalized on the hydro grid it is sampled on (<Liquefier><normalize_on_hydro_grid>, default on).
        """
    def get_source(self, tau: typing.SupportsFloat | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, eta: typing.SupportsFloat | typing.SupportsIndex) -> typing.Annotated[list[float], "FixedSize(4)"]:
        """
        Summed source current j^mu [GeV/fm^4] at a Milne point, causality-gated. This is the reference the pure-Python kernel is checked against.
        """
    def normalizable_on_grid(self) -> bool:
        ...
    def normalization_summary(self) -> str:
        ...
    def query_time(self, tau: typing.SupportsFloat | typing.SupportsIndex, dtau: typing.SupportsFloat | typing.SupportsIndex, k: typing.SupportsInt | typing.SupportsIndex) -> float:
        ...
    def set_hydro_step(self, dtau: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_late_hydro_step(self, dtau_late: typing.SupportsFloat | typing.SupportsIndex, tau_switch: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_normalize_on_hydro_grid(self, on: bool) -> None:
        ...
    def smearing_kernel(self, tau: typing.SupportsFloat | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, eta: typing.SupportsFloat | typing.SupportsIndex, droplet: Droplet) -> list[float]:
        ...
class CausalLiquefier(LiquefierBase):
    """
    Causal (telegraph) diffusion of deposited energy-momentum.
    The default constructor reads <Liquefier><CausalLiquefier> from the loaded JETSCAPE XML; the 4-argument one takes the grid directly and needs no XML.
    """
    def InitializeParameters(self) -> None:
        """
        Re-read the XML block and recompute c_diff / gamma_relax.
        """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, dtau: typing.SupportsFloat | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, deta: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def get_peta(self, px: typing.SupportsFloat | typing.SupportsIndex, pz: typing.SupportsFloat | typing.SupportsIndex, eta: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def get_ptau(self, px: typing.SupportsFloat | typing.SupportsIndex, pz: typing.SupportsFloat | typing.SupportsIndex, eta: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def j_delta(self, t: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def j_smooth(self, t: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def kernel_j(self, t: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def kernel_rho(self, t: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def params(self) -> dict:
        """
        The five parameters fast_data.liquefier.LiquefierParams needs, read from the live C++ object rather than duplicated on the Python side.
        """
    def rho_delta(self, t: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def rho_smooth(self, t: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def set_t_delay(self, tau_delay: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def c_diff(self) -> float:
        ...
    @property
    def d_diff(self) -> float:
        ...
    @d_diff.setter
    def d_diff(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def deta(self) -> float:
        ...
    @deta.setter
    def deta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def dtau(self) -> float:
        ...
    @dtau.setter
    def dtau(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def dx(self) -> float:
        ...
    @dx.setter
    def dx(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def dy(self) -> float:
        ...
    @dy.setter
    def dy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gamma_relax(self) -> float:
        ...
    @property
    def tau_delay(self) -> float:
        ...
    @tau_delay.setter
    def tau_delay(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def time_relax(self) -> float:
        ...
    @time_relax.setter
    def time_relax(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def width_delta(self) -> float:
        ...
    @width_delta.setter
    def width_delta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MpiMusic(FluidDynamics):
    """
            MUSIC viscous relativistic hydrodynamics module.
    
            Registered module name: ``"MUSIC"`` — instantiate via::
    
                hydro = create_module("MUSIC")
    
            Important setup before calling Init():
                * The JetScape XML must have ``<Hydro><MUSIC>`` settings with
                  ``output_evolution_to_memory: 1`` for bulk_info to be populated.
                * Call ``hydro.set_preserve_bulk_info(True)`` so that bulk_info
                  survives the ClearTasks() call at the end of each event.
    
            After Exec() completes, ``hydro.get_bulk_info()`` returns an
            EvolutionHistory with all stored fluid cells.
          
    """
    def GetHydroFreezeOutTemperature(self) -> float:
        """
        Return the freeze-out temperature set in the XML [GeV].
        """
    def clear_hydro_info_from_memory(self) -> None:
        """
        Release MUSIC's native evolution store.  FastRootBulkWriter does this after writing each event.
        """
    def get_checkpoint_tau(self) -> float:
        """
        tau [fm/c] by which this instance keeps a checkpoint of its state in every evolution (0: off).
        """
    def get_dump_hydro_only(self) -> bool:
        """
        Return whether <dump_hydro_only> is enabled (native store kept, bulk_info.data not built).
        """
    def get_freeze_out_surface(self) -> bool:
        """
        Whether MUSIC builds the freeze-out surface for this instance.
        """
    def get_hit_grid_boundary(self) -> bool:
        """
        True if the last evolution stopped because the freeze-out surface reached the transverse grid boundary (MUSIC's reRunHydro): the stored evolution is truncated.  Reset at the start of every evolution.
        """
    def get_native_evolution_numpy(self, tau_stride: typing.SupportsInt | typing.SupportsIndex = 1) -> numpy.typing.NDArray[numpy.float32]:
        """
                     Copy MUSIC's native evolution store into a numpy array in one C++
                     pass (GIL released), without building bulk_info.data.
        
                     The values and layout are exactly what FastRootBulkWriter writes
                     in ``native`` mode, so ``arr.ravel()`` equals one ``user_res``
                     entry of its ROOT output.
        
                     Requires ``<Hydro><MUSIC><dump_hydro_only>1``.  If a
                     FastRootBulkWriter is in the pipeline it releases the store at the
                     end of its Exec(), so read the store before that runs.
        
                     Parameters
                     ----------
                     tau_stride : int, default 1
                         Keep every N-th stored tau step.
        
                     Returns
                     -------
                     np.ndarray, shape (ntau, nx, ny, neta, 4), dtype float32
                         Features: [energy_density, vx, vy, vz].  Grid spacing and
                         origin are in ``get_bulk_info()`` (dtau is multiplied by
                         ``tau_stride``).
        """
    def get_native_fluid_cell(self, idx: typing.SupportsInt | typing.SupportsIndex) -> FluidCellInfo:
        """
                     Return one cell of MUSIC's native store as a FluidCellInfo.
        
                     ``idx`` is the flat tau-major (tau, x, y, eta) index, i.e.
                     ``((it*nx + ix)*ny + iy)*neta + ieta`` with the grid sizes from
                     ``get_bulk_info()``.
        """
    def get_native_frame_numpy(self, it: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float32]:
        """
        One tau step of MUSIC's native store, (nx, ny, neta, 4) float32.
        """
    def get_native_ntau(self) -> int:
        """
        Number of tau steps in MUSIC's native store (0 when empty).
        """
    def get_number_of_fluid_cells(self) -> int:
        """
        Return the number of cells in MUSIC's native evolution store (0 before InitializeHydro() or after the store was released).
        """
    def get_preserve_bulk_info(self) -> bool:
        """
        Return whether bulk_info preservation is enabled.
        """
    def get_restart_status(self) -> str:
        """
        What the last evolution did with set_restart_from ('' if not set).
        """
    def get_skip_surface(self) -> bool:
        """
        Return whether <skip_surface> is enabled (freeze-out surface not exported to the framework).
        """
    def get_slim_bulk_info(self) -> bool:
        """
        Return whether <slim_bulk_info> is on: bulk_info gets a slim copy (e, s, T, vx, vy, vz) instead of full FluidCellInfo cells.
        """
    def get_store_evolution(self) -> bool:
        """
        Whether MUSIC stores this instance's evolution frames in memory.
        """
    def set_checkpoint_tau(self, tau: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Keep a checkpoint of this instance's state in every evolution, at the last freeze-out step whose RK step ends by tau [fm/c] (0: off), for another instance to start from (set_restart_from).  music4gpu; CPU MUSIC ignores it.
        """
    def set_dump_hydro_only(self, dump: bool) -> None:
        """
        Keep this instance's native evolution store and skip building bulk_info.data.  Call it AFTER JetScape.Init(): InitializeHydro() resets the flag from the XML.  Needs output_evolution_to_memory = 1. Nothing may query this instance's medium (energy loss reads the first hydro only).
        """
    def set_freeze_out_surface(self, build: bool) -> None:
        """
        Build the freeze-out surface (True) or not (False) for this instance, from its next evolution on.  Without a surface MUSIC stops on the equivalent max(e) < e_fo test at the same step (music4gpu; CPU MUSIC always builds it).  The XML default is <Hydro><MUSIC><freeze_out_surface> in the first block, overridden per instance in its own block.
        """
    def set_preserve_bulk_info(self, preserve: bool) -> None:
        """
        When True, bulk_info.data is retained after Clear() so that it can be inspected from Python after the event finishes.  Set this BEFORE calling Init().
        """
    def set_restart_from(self, other: MpiMusic) -> None:
        """
        Start every evolution of this instance from `other`'s latest checkpoint when it fits (same parameters, initial arena and initial-state source, no droplet before its step), from tau0 otherwise: the output is the same either way.  None: always from tau0.  MUSIC_RESTART_VERIFY=1 runs from tau0 and compares the state with the checkpoint at its step.  music4gpu; CPU MUSIC ignores it.
        """
    def set_skip_surface(self, skip: bool) -> None:
        """
        Do not export this instance's freeze-out surface to the framework. Call it AFTER JetScape.Init(), like set_dump_hydro_only.
        """
    def set_slim_bulk_info(self, slim: bool) -> None:
        """
        Override <slim_bulk_info> for this instance. Call it AFTER JetScape.Init() (which reads the XML flag).
        """
    def set_store_evolution(self, store: bool) -> None:
        """
        Store this instance's evolution frames in MUSIC's in-memory store (True, default) or not (False), from its next evolution on.  False keeps the grid metadata and the frame count (bulk_info, get_native_ntau) but no cells: for an instance whose evolution nobody reads.  music4gpu; CPU MUSIC ignores it.
        """
class TrentoInitial(InitialState):
    """
            TRENTo initial-condition module.
    
            Registered module name: ``"TrentoInitial"`` — instantiate via::
    
                ini = create_module("TrentoInitial")
    
            After Exec() completes:
                * ``ini.get_entropy_density_numpy()`` returns the entropy-density
                  grid as a 2D numpy array.
                * ``ini.get_event_info()`` returns a dict of per-event geometric
                  quantities (impact parameter, Npart, Ncoll, eccentricities …).
          
    """
    def GetEventCentrality(self) -> float:
        """
        Return the event centrality percentile [0–100].
        """
    def get_eccentricity(self, order: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Return the n-th order participant-plane eccentricity epsilon_n.
        """
    def get_event_centrality(self) -> float:
        """
        Return the event centrality percentile [0–100].
        """
    def get_event_info(self) -> dict:
        """
                     Return per-event geometric information as a Python dict.
        
                     Keys
                     ----
                     impact_parameter, num_participant, num_binary_collisions,
                     total_entropy, normalization, event_centrality, xmid, ymid,
                     eccentricities (dict {order: epsilon_n}),
                     participant_planes (dict {order: psi_n}).
        """
    def get_impact_parameter(self) -> float:
        """
        Return the impact parameter b [fm].
        """
    def get_num_binary_collisions(self) -> float:
        """
        Return the number of binary collisions (Ncoll).
        """
    def get_num_participant(self) -> float:
        """
        Return the number of wounded nucleons (Npart).
        """
    def get_participant_plane_angle(self, order: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Return the n-th order participant-plane angle psi_n [rad].
        """
    def get_total_entropy(self) -> float:
        """
        Return the total entropy of the event.
        """
class FastRootBulkWriter(JetScapeModuleBase):
    """
            Hydro-only ROOT writer for the MUSIC evolution.
    
            Reads MUSIC's native in-memory store directly and never builds the
            framework bulk_info.data, so it is much faster and lighter than
            RootBulkWriter / PyBulkRootWriter.
    
            Registered module name: ``"FastRootBulkWriter"`` — instantiate via::
    
                writer = create_module("FastRootBulkWriter")
    
            or let the XML task list create it from a top-level
            ``<FastRootBulkWriter>`` block.
    
            Requirements:
                * ``<Hydro><MUSIC><dump_hydro_only>1`` and
                  ``<output_evolution_to_memory>1`` (MUSIC is the only supported
                  hydro).
                * Add it to the pipeline AFTER the hydro module.
                * Settings come from the ``<FastRootBulkWriter>`` XML block
                  (out_file_name, grid_mode = native|grid, tau_stride, and the
                  user grid for grid mode).
    
            The ROOT file is written and closed by ``Finish()`` (called by
            ``JetScape.Finish()``).  Read it back with
            ``jetscape.fast_root_bulk.read_fast_root_bulk()``.
          
    """
    def GetDtau(self) -> float:
        """
        Return the written tau step (includes tau_stride) [fm/c].
        """
    def GetGridMode(self) -> str:
        """
        Return the grid mode: "native" (MUSIC grid) or "grid" (user grid).
        """
    def GetNFeatures(self) -> int:
        """
        Return the number of features per cell (4: energy_density, vx, vy, vz).
        """
    def GetNeta(self) -> int:
        ...
    def GetNtauWritten(self) -> int:
        """
        Return the number of tau steps in the last written event.
        """
    def GetNumberOfEventsWritten(self) -> int:
        """
        Return the number of events filled into the tree so far.
        """
    def GetNx(self) -> int:
        ...
    def GetNy(self) -> int:
        ...
    def GetOutFileName(self) -> str:
        """
        Return the output ROOT file name (<out_file_name>).
        """
    def GetTauFreezeout(self) -> float:
        """
        Return tau one step past the last stored MUSIC step [fm/c].
        """
    def GetTauMin(self) -> float:
        """
        Return the tau of the first written step [fm/c].
        """
    def GetTauStride(self) -> int:
        """
        Return the native-mode tau stride (every N-th stored step).
        """
    def IsFileOpen(self) -> bool:
        """
        Return True while the ROOT file is open (between the first event and Finish()).
        """
    def get_event_layout(self) -> dict:
        """
                     Return the writer state as a dict.
        
                     Keys
                     ----
                     out_file_name, grid_mode, tau_stride, events_written,
                     shape (ntau, nx, ny, neta, nFeatures) of the last written event,
                     tau_min, dtau, tau_freezeout.
        """
class Vertex:
    """
    A splitting vertex: space-time position (x,y,z) [fm], t [fm/c]. Vertex(x, y, z, t) or Vertex(FourVector).
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex, t: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, x: FourVector) -> None:
        ...
    def set_location(self, x: FourVector) -> None:
        ...
    def t(self) -> float:
        ...
    def x(self) -> float:
        ...
    def x_in(self) -> FourVector:
        ...
    def y(self) -> float:
        ...
    def z(self) -> float:
        ...
class PartonShower:
    """
    Parton shower as a directed graph (vertices = nodes, partons = edges); ids as in to_numpy() / vertices_to_numpy().
    """
    @property
    def nodes(self) -> 'list[ShowerNode]':
        """
        Every vertex as a ShowerNode, in vertices_to_numpy order.
        """
    def __init__(self) -> None:
        ...
    def __iter__(self) -> 'Iterator[Parton]':
        """
        The partons (edges) of the shower.
        """
    def children(self, parton: Parton) -> list:
        """
        The partons coming out of the vertex where this parton ends.
        """
    def edge_array(self) -> np.ndarray:
        """
        The partons as EDGE_DTYPE records (src, tgt, pid, pstat, E px py pz, t x y z).
        """
    def edge_of(self, parton: Parton) -> tuple[int, int, int]:
        """
        (edge id, source vertex id, target vertex id) of a parton of this shower.
        """
    def end_node(self, parton: 'Parton') -> 'ShowerNode':
        """
        The ShowerNode where `parton` ends.
        """
    def end_vertex(self, parton: Parton) -> Vertex:
        ...
    def final_partons(self) -> list:
        """
        The partons at the end of the shower.
        """
    def is_end_vertex(self, vertex_id: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    def is_one_to_two(self, vertex_id: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    def new_parton(self, source_id: typing.SupportsInt | typing.SupportsIndex, target_id: typing.SupportsInt | typing.SupportsIndex, parton: Parton) -> int:
        """
        Add a parton from vertex source_id to vertex target_id.
        """
    def new_vertex(self, vertex: Vertex) -> int:
        """
        Add a vertex; returns its id.
        """
    def node_array(self) -> np.ndarray:
        """
        The vertices as NODE_DTYPE records (id, t, x, y, z).
        """
    def number_of_partons(self) -> int:
        ...
    def number_of_vertices(self) -> int:
        ...
    def parents(self, parton: Parton) -> list:
        """
        The partons going into the vertex where this parton starts.
        """
    def parton_at(self, i: typing.SupportsInt | typing.SupportsIndex) -> Parton:
        """
        Parton on the i-th edge.
        """
    def partons(self) -> list:
        """
        Every parton (edge) of the shower.
        """
    def production_vertex(self, parton: Parton) -> Vertex:
        ...
    def split_delta_r(self, vertex_id: typing.SupportsInt | typing.SupportsIndex) -> float:
        ...
    def split_kt(self, vertex_id: typing.SupportsInt | typing.SupportsIndex) -> float:
        ...
    def split_z(self, vertex_id: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        z of the 1 -> 2 split at vertex id.
        """
    def start_node(self, parton: 'Parton') -> 'ShowerNode':
        """
        The ShowerNode where `parton` starts.
        """
    def to_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
                     Flatten the shower's edges to a numpy float64 array, one row per
                     parton, shape (n_partons, 12):
        
                       [source_id, target_id, pid, pstat, px, py, pz, E, x, y, z, t]
        
                     Momentum (px,py,pz,E) in GeV; position (x,y,z) in fm; t in fm/c.
                     (x,y,z,t) is the parton's production position (source vertex);
                     use source_id / target_id to correlate with vertices_to_numpy().
        """
    def vertex_at(self, i: typing.SupportsInt | typing.SupportsIndex) -> Vertex:
        """
        Vertex on the i-th node.
        """
    def vertex_by_id(self, id: typing.SupportsInt | typing.SupportsIndex) -> Vertex:
        ...
    def vertex_id(self, vertex: Vertex) -> int:
        ...
    def vertices_to_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
                     Flatten the shower's nodes to a numpy float64 array, one row per
                     splitting vertex, shape (n_vertices, 5):
        
                       [node_id, x, y, z, t]
        
                     Position (x,y,z) in fm; t in fm/c.  node_id matches the
                     source_id / target_id columns in to_numpy(), allowing the two
                     arrays to be joined into a directed graph (see
                     jetscape.utils.shower_to_networkx).
        """
class JetEnergyLoss(JetScapeModuleBase):
    """
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
          
    """
    def AddPartonShowerGenerator(self, generator: typing.Any) -> None:
        """
        Set how this task grows its shower (PartonShowerGeneratorDefault unless set).
        """
    def AddShowerInitiatingParton(self, parton: Parton) -> None:
        """
        Set the parton this task showers (the manager does this per shower).
        """
    def Clone(self) -> JetEnergyLoss:
        """
        A copy for one shower (C++ modules: their copy constructor).
        """
    def DoEnergyLoss(self, deltaT: typing.SupportsFloat | typing.SupportsIndex, time: typing.SupportsFloat | typing.SupportsIndex, Q2: typing.SupportsFloat | typing.SupportsIndex, pIn: list, pOut: list) -> None:
        """
        One step: pIn (list of Parton, changed in place), pOut (list, filled with the partons at time + deltaT).
        """
    def GetDeltaT(self) -> float:
        ...
    def GetGetHydroCellSignalConnected(self) -> bool:
        ...
    def GetHydroCell(self, t: typing.SupportsFloat | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> typing.Any:
        """
        The medium at (t, x, y, z) [fm] as a FluidCellInfo, or None without a hydro (the GetHydroCell signal Matter and LBT use).
        """
    def GetMaxT(self) -> float:
        ...
    def GetPartonShowerGenerator(self) -> typing.Any:
        ...
    def GetQhat(self) -> float:
        ...
    def GetSentInPartonsConnected(self) -> bool:
        ...
    def GetShower(self) -> PartonShower:
        """
        The finished PartonShower for this task (valid after Exec).
        """
    def GetShowerInitiatingParton(self) -> Parton:
        """
        The parton this shower started from (the per-shower copies).
        """
    def GetStartT(self) -> float:
        ...
    def GetUseIntialPartonShower(self) -> bool:
        ...
    def SetDeltaT(self, deltaT: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetMaxT(self, maxT: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetQhat(self, qhat: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetStartT(self, startT: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetUseIntialPartonShower(self, use: bool) -> None:
        """
        Continue a given shower (ISR) instead of starting from one parton.
        """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, other: JetEnergyLoss) -> None:
        """
        Copy (the C++ copy constructor: id, settings, a new task number).
        """
    def add_a_liquefier(self, liquefier: typing.Any) -> None:
        """
        Attach the liquefier that collects droplets from this shower. The framework calls liquefier->add_hydro_sources() centrally in DoExecTime(), so Matter and LBT need no changes.
        """
    def get_liquefier(self) -> LiquefierBase:
        """
        The attached liquefier, or None.
        """
class JetEnergyLossManager(JetScapeModuleBase):
    """
    Energy-loss manager; one JetEnergyLoss child per shower-initiating parton.
    """
    @property
    def showers(self) -> 'list[PartonShower]':
        """
        The PartonShowers of this event, one per shower (get_showers()).
        """
    def __init__(self) -> None:
        ...
    def final_parton_array(self) -> np.ndarray:
        """
        The final partons as FINAL_PARTON_DTYPE records (final_partons_numpy).  After the
            manager's Exec, before ClearPerEvent.
        """
    def final_partons_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        The final-state partons the hadronization would receive, as an (N, 14) float64 array with columns FINAL_PARTON_COLUMNS (shower index = position in the manager's task list). Valid after the manager's Exec, before ClearPerEvent.
        """
    def get_shower_initiating_partons(self) -> list[Parton]:
        """
        The hard (shower-initiating) partons, one per shower.
        """
    def get_showers(self) -> list[PartonShower]:
        """
        List of per-event PartonShowers (one per shower-initiating parton). Valid after the manager's Exec, before ClearPerEvent.
        """
    def initiator_array(self) -> np.ndarray:
        """
        The shower-initiating partons as INITIATOR_DTYPE records
            (showers.initiators_from_manager).
        """
    def number_of_showers(self) -> int:
        ...
class HardProcess(JetScapeModuleBase):
    """
            The hard process of an event (PythiaGun, PGun, ... in C++).
    
            A Python gun subclasses it and fills the event in ExecuteTask() with
            AddParton(Parton(...)) (and AddHadron() for hadrons from the hard
            process); GetSigmaGen, GetSigmaErr, GetPtHat and GetEventWeight can be
            overridden for the writers. The hard vertex comes from the initial state:
            JetScapeSignalManager.Instance().GetInitialStatePointer()
            .sample_binary_collision_point(). register_module("CustomModuleX", cls)
            makes it available to the XML (<Hard><CustomModuleX>).
          
    """
    @property
    def info(self) -> dict:
        """
        hard_process_info(): cross section, weight, pTHat, ... of this event.
        """
    def AddHadron(self, hadron: Hadron) -> None:
        ...
    def AddParton(self, parton: Parton) -> None:
        ...
    def AddPartonShower(self, shower: PartonShower) -> None:
        ...
    def GetEventWeight(self) -> float:
        ...
    def GetHadronList(self) -> list:
        ...
    def GetMax_Color(self) -> float:
        ...
    def GetMax_ColorPerShower(self) -> float:
        ...
    def GetNHadrons(self) -> int:
        ...
    def GetNHardPartons(self) -> int:
        ...
    def GetNPartonShowers(self) -> int:
        ...
    def GetPartonAt(self, i: typing.SupportsInt | typing.SupportsIndex) -> Parton:
        ...
    def GetPartonList(self) -> list:
        """
        The hard partons of the event (the objects, not copies).
        """
    def GetPartonShowerAt(self, i: typing.SupportsInt | typing.SupportsIndex) -> PartonShower:
        ...
    def GetPtHat(self) -> float:
        ...
    def GetRemnants(self) -> list[Parton]:
        """
        The beam remnants (copies), with ISR.
        """
    def GetSigmaErr(self) -> float:
        ...
    def GetSigmaGen(self) -> float:
        ...
    def PushRemnants(self, parton: Parton) -> None:
        ...
    def SetMax_Color(self, col: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def __init__(self) -> None:
        ...
    def parton_array(self) -> np.ndarray:
        """
        The hard partons as HARD_PARTON_DTYPE records (pid, E px py pz, pt, y), hardest
            first (hard_partons_numpy).
        """
class Hadronization(JetScapeModuleBase):
    """
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
          
    """
    @property
    def last_random_seed(self) -> int:
        """
        ColorlessHadronization's (or iColoredHadronization's) last seed.
        """
    def DeleteHadrons(self) -> None:
        ...
    def DeleteRealHadrons(self) -> None:
        ...
    def DoHadronization(self, shower: list, hOut: list, pOut: list) -> None:
        """
        Hadronize: shower (list of lists of Parton), hOut and pOut (lists, filled with the hadrons and the partons left over).
        """
    def GetHadrons(self) -> list:
        """
        The hadrons this task made in the event.
        """
    def GetHydroCell(self, t: typing.SupportsFloat | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex) -> typing.Any:
        ...
    def GetHydroHyperSurface(self, T: typing.SupportsFloat | typing.SupportsIndex) -> list:
        """
        The hydro's constant-temperature surface at T [GeV] as SurfaceCellInfos (empty without a hydro).
        """
    def GetOutPartons(self) -> list:
        """
        The partons this task left over in the event.
        """
    def __init__(self) -> None:
        ...
    def colour_audit(self) -> dict:
        """
        iColoredHadronization: the last event's colour audit (a dict).
        """
    def colour_totals(self) -> dict:
        """
        iColoredHadronization: the audits added up over the run.
        """
    def hadron_array(self) -> HadronArray:
        """
        The hadrons this module (or every module of the manager) made this event
            (hadronization_hadrons_numpy) as a HadronArray (one sample).
        """
    def hadronize(self, partons, seed: int | None = None) -> HadronArray:
        """
        Hadronize stored final partons: FINAL_PARTON_DTYPE records (final_parton_array())
            or the old (N, 14) array (hadronize_partons).  -> HadronArray; info holds the seed
            used and n_partons_out.
        """
    def set_colour_closure(self, mode: str) -> None:
        """
        iColoredHadronization: <colour_closure> by hand (reconnect, reconnect_global,
            thermal, fake, none).
        """
class HadronizationManager(JetScapeModuleBase):
    """
    The jet hadronization of an event: it takes the energy-loss manager's final partons and runs its Hadronization tasks (Add() a Hadronization with the modules in it).
    """
    @property
    def last_random_seed(self) -> int:
        """
        ColorlessHadronization's (or iColoredHadronization's) last seed.
        """
    def DeleteHadrons(self) -> None:
        ...
    def DeleteRealHadrons(self) -> None:
        ...
    def GetHadrons(self) -> list:
        """
        The hadrons of all its tasks in this event.
        """
    def __init__(self) -> None:
        ...
    def colour_audit(self) -> dict:
        """
        iColoredHadronization: the last event's colour audit (a dict).
        """
    def colour_totals(self) -> dict:
        """
        iColoredHadronization: the audits added up over the run.
        """
    def hadron_array(self) -> HadronArray:
        """
        The hadrons this module (or every module of the manager) made this event
            (hadronization_hadrons_numpy) as a HadronArray (one sample).
        """
    def hadronize(self, partons, seed: int | None = None) -> HadronArray:
        """
        Hadronize stored final partons: FINAL_PARTON_DTYPE records (final_parton_array())
            or the old (N, 14) array (hadronize_partons).  -> HadronArray; info holds the seed
            used and n_partons_out.
        """
    def set_colour_closure(self, mode: str) -> None:
        """
        iColoredHadronization: <colour_closure> by hand (reconnect, reconnect_global,
            thermal, fake, none).
        """
class SoftParticlization(JetScapeModuleBase):
    """
            Particlization of the hydro's freeze-out surface (iSS in C++).
    
            A Python sampler subclasses it, gets the surface in ExecuteTask() with
            GetHydroHyperSurface() (or FindHydroHyperSurface() from the stored
            evolution) and stores the event's hadrons with set_hadron_list(samples),
            one list of Hadrons per oversample. register_module("CustomModuleX", cls)
            and <SoftParticlization><CustomModuleX> make the XML use it.
          
    """
    @property
    def last_random_seed(self) -> int:
        """
        The seed used for the last event (soft_last_random_seed).
        """
    def ClearHydroHyperSurface(self) -> None:
        ...
    def GetHydroHyperSurface(self) -> list:
        """
        The hydro's freeze-out surface (its own, as MUSIC gives it) as SurfaceCellInfos; empty without one.
        """
    def GetLastRandomSeed(self) -> int:
        ...
    def SetNextRandomSeed(self, seed: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def SetNumberOfSamples(self, n: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    def __init__(self) -> None:
        ...
    def get_hadron_list(self) -> list:
        """
        The event's hadrons: one list of Hadrons per oversample.
        """
    def hadron_array(self) -> HadronArray:
        """
        This event's hadrons (soft_hadrons_numpy) as a HadronArray, one sample per
            oversample.  With use_compact_output() the arrays are taken over without a copy:
            call it once per event.
        """
    def primary_array(self) -> 'HadronArray | None':
        """
        The undecayed hadrons of the first N oversamples (soft_primaries_numpy) as a
            HadronArray; None if none were kept.  Once per event, after hadron_array().
        """
    def sample_cells(self, cells, seed: int, n_samples: int) -> HadronArray:
        """
        Sample the given (N, 32) surface cells now (soft_sample_cells) -> HadronArray.
        """
    def set_hadron_list(self, samples: list) -> None:
        """
        Store the event's hadrons: a list of oversamples, each a list of Hadrons.
        """
    def set_next_random_seed(self, seed: int) -> None:
        """
        Seed for the next event's sampling, one-shot (soft_set_next_random_seed).
        """
    def set_number_of_samples(self, n: int) -> None:
        """
        Oversamples per event from the next event on (soft_set_number_of_samples).
        """
    def use_compact_output(self, on: bool = True) -> None:
        """
        Hand the hadrons over as arrays from the next event on (soft_set_compact_output).
        """
class Afterburner(JetScapeModuleBase):
    """
            Hadronic transport after particlization (SMASH in C++).
    
            A Python afterburner subclasses it, takes its input in ExecuteTask() with
            GatherAfterburnerHadrons() (one list per oversample: the soft hadrons plus
            the jet hadrons), propagates them, and returns the result from
            GetCurrentHadronList(). register_module("CustomModuleX", cls) and
            <Afterburner><CustomModuleX> make the XML use it.
          
    """
    def GatherAfterburnerHadrons(self) -> list:
        """
        The hadrons for the transport: per oversample, the soft and the jet hadrons.
        """
    def GetCurrentHadronList(self) -> list[Hadron]:
        ...
    def GetFragmentationHadrons(self) -> list:
        ...
    def GetSoftParticlizationHadrons(self) -> list:
        ...
    def __init__(self) -> None:
        ...
class JetScapeEventHeader:
    """
    What the tasks report for an event (CollectHeader): weights, vertex, centrality.
    """
    def GetEventCentrality(self) -> float:
        ...
    def GetEventPlaneAngle(self) -> float:
        ...
    def GetEventWeight(self) -> float:
        ...
    def GetNcoll(self) -> float:
        ...
    def GetNpart(self) -> float:
        ...
    def GetPtHat(self) -> float:
        ...
    def GetSigmaErr(self) -> float:
        ...
    def GetSigmaGen(self) -> float:
        ...
    def GetTotalEntropy(self) -> float:
        ...
    def GetVertexX(self) -> float:
        ...
    def GetVertexY(self) -> float:
        ...
    def GetVertexZ(self) -> float:
        ...
    def SetEventCentrality(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetEventPlaneAngle(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetEventWeight(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetNcoll(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetNpart(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetPtHat(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetSigmaErr(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetSigmaGen(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetTotalEntropy(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetVertexX(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetVertexY(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def SetVertexZ(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class JetScapeWriter(JetScapeModuleBase):
    """
            An output writer (ASCII, HepMC, ... in C++).
    
            A Python writer subclasses it, implements GetStatus() (True when ready),
            and overrides Write(obj): it gets every Parton, Vertex, PartonShower,
            Hadron and str the tasks write in an event, after WriteHeaderToFile()
            (GetHeader() then holds the event header) and before WriteEvent(). Add
            it to the pipeline with JetScape.Add(writer).
          
    """
    def Close(self) -> None:
        ...
    def GetHeader(self) -> JetScapeEventHeader:
        ...
    def GetOutputFileName(self) -> str:
        ...
    def GetStatus(self) -> bool:
        ...
    def Open(self) -> None:
        ...
    def SetOutputFileName(self, file_name: str) -> None:
        ...
    @typing.overload
    def Write(self, obj: Parton) -> None:
        ...
    @typing.overload
    def Write(self, obj: Hadron) -> None:
        ...
    @typing.overload
    def Write(self, obj: Vertex) -> None:
        ...
    @typing.overload
    def Write(self, obj: PartonShower) -> None:
        ...
    @typing.overload
    def Write(self, obj: str) -> None:
        ...
    def WriteComment(self, s: str) -> None:
        ...
    def WriteEvent(self) -> None:
        ...
    def WriteHeaderToFile(self) -> None:
        ...
    def WriteWhiteSpace(self, s: str) -> None:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, file_name: str) -> None:
        ...
class PartonShowerGenerator:
    """
    How a JetEnergyLoss task grows its shower (DoShower(jloss), and the per-time-step hooks). Subclass it in Python and hand it to JetEnergyLoss.AddPartonShowerGenerator().
    """
    def DoCalculateTime(self, jloss: JetEnergyLoss) -> None:
        ...
    def DoExecTime(self, jloss: JetEnergyLoss) -> None:
        ...
    def DoFinishPerEvent(self, jloss: JetEnergyLoss) -> None:
        ...
    def DoInitPerEvent(self, jloss: JetEnergyLoss) -> None:
        ...
    def DoShower(self, jloss: JetEnergyLoss) -> None:
        ...
    def __init__(self) -> None:
        ...
class PartonShowerGeneratorDefault(PartonShowerGenerator):
    """
    The final-state shower generator every JetEnergyLoss task uses by default.
    """
    def __init__(self) -> None:
        ...
class IsrShowerPSG(PartonShowerGeneratorDefault):
    """
    The initial-state (backward) shower generator of IsrJet.
    """
    def __init__(self) -> None:
        ...
class IsrManager(JetEnergyLossManager):
    """
            The initial-state radiation of the hard partons: an energy-loss manager run
            backward in time under PythiaIsrGun. As the XML builds it for <Hard><ISR>:
    
                isr = IsrManager(); jet = IsrJet(); imatter = iMATTER()
                jet.SetDeltaT(-dt); jet.SetStartT(0); jet.SetMaxT(-maxT); imatter.SetMaxT(-maxT)
                jet.AddPartonShowerGenerator(IsrShowerPSG()); jet.Add(imatter)
                isr.Add(jet); gun = create_module("PythiaIsrGun"); gun.Add(isr)
                gun.Add(create_module("MCGlauberGenString"))
          
    """
    def __init__(self) -> None:
        ...
class IsrJet(JetEnergyLoss):
    """
    The per-parton task of the IsrManager (an initial-state shower).
    """
    def __init__(self) -> None:
        ...
class iMATTER(JetEnergyLoss):
    """
    MATTER's initial-state (space-like) shower.
    """
    def __init__(self) -> None:
        ...
class JetScapeReaderAscii:
    """
    Read a JetScapeWriterAscii file: while not r.Finished(): r.Next(); r.GetPartonShowers(), r.GetHadrons(), r.GetSigmaGen(), ...
    """
    def Close(self) -> None:
        ...
    def Finished(self) -> bool:
        """
        True at the end of the file.
        """
    def GetCurrentEvent(self) -> int:
        ...
    def GetCurrentNumberOfPartonShowers(self) -> int:
        ...
    def GetEventPlaneAngle(self) -> float:
        ...
    def GetEventWeight(self) -> float:
        ...
    def GetHadrons(self) -> list:
        ...
    def GetPartonShowers(self) -> list[PartonShower]:
        ...
    def GetSigmaErr(self) -> float:
        ...
    def GetSigmaGen(self) -> float:
        ...
    def GetVertexX(self) -> float:
        ...
    def GetVertexY(self) -> float:
        ...
    def GetVertexZ(self) -> float:
        ...
    def Next(self) -> None:
        """
        Read the next event.
        """
    def __init__(self, file_name: str) -> None:
        ...
class JetScapeReaderAsciiGZ:
    """
    As JetScapeReaderAscii, for .gz files.
    """
    def Close(self) -> None:
        ...
    def Finished(self) -> bool:
        """
        True at the end of the file.
        """
    def GetCurrentEvent(self) -> int:
        ...
    def GetCurrentNumberOfPartonShowers(self) -> int:
        ...
    def GetEventPlaneAngle(self) -> float:
        ...
    def GetEventWeight(self) -> float:
        ...
    def GetHadrons(self) -> list:
        ...
    def GetPartonShowers(self) -> list[PartonShower]:
        ...
    def GetSigmaErr(self) -> float:
        ...
    def GetSigmaGen(self) -> float:
        ...
    def GetVertexX(self) -> float:
        ...
    def GetVertexY(self) -> float:
        ...
    def GetVertexZ(self) -> float:
        ...
    def Next(self) -> None:
        """
        Read the next event.
        """
    def __init__(self, file_name: str) -> None:
        ...
class JetScapeSignalManager:
    """
            Singleton signal/pointer manager for the JETSCAPE framework.
    
            Provides global read access to all registered top-level physics
            modules so that any module (including Python modules added to the
            pipeline via JetScape.Add()) can look up its peers by type without
            holding an explicit reference.
    
            The singleton is populated automatically by JetScape::Init() for all
            modules registered with JetScape.Add().  Python code that creates a
            custom module (e.g. PyBulkRootWriter) can therefore access any
            framework module from inside its Exec() method:
    
                from pyjetscape import JetScapeSignalManager
                sm    = JetScapeSignalManager.Instance()
                hydro = sm.GetHydroPointer()   # FluidDynamics or None
                if hydro is not None:
                    info = hydro.get_bulk_info()
    
            Notes
            -----
            * The object returned by Instance() is owned by the C++ singleton; do
              not store it across events or after Finish().
            * All Get*Pointer() methods lock the stored weak_ptr and return the
              shared_ptr.  They return None if the module was never registered.
          
    """
    @staticmethod
    def Instance() -> JetScapeSignalManager:
        """
        Return the singleton JetScapeSignalManager instance.
        """
    def GetHydroPointer(self) -> FluidDynamics:
        """
        Return the registered FluidDynamics module, or None if not set.
        """
    def GetInitialStatePointer(self) -> InitialState:
        """
        Return the registered InitialState module, or None if not set.
        """
    def GetJetEnergyLossManagerPointer(self) -> JetEnergyLossManager:
        """
        Return the registered JetEnergyLossManager (holds the per-event parton showers), or None if not set.
        """
    def GetNumberOfEdensitySignals(self) -> int:
        """
        Return the number of connected energy-density signals.
        """
    def GetNumberOfGetHydroCellSignals(self) -> int:
        """
        Return the number of connected GetHydroCell signals.
        """
    def GetNumberOfJetSignals(self) -> int:
        """
        Return the number of connected jet signals.
        """
    def GetPreEquilibriumPointer(self) -> PreequilibriumDynamics:
        """
        Return the registered PreequilibriumDynamics module, or None if not set.
        """
    def SetHydroPointer(self, hydro: typing.Any) -> None:
        """
        Register a FluidDynamics module with the signal manager.
        """
    def SetInitialStatePointer(self, ini: typing.Any) -> None:
        """
        Register an InitialState module with the signal manager.
        """
    def SetPreEquilibriumPointer(self, preeq: typing.Any) -> None:
        """
        Register a PreequilibriumDynamics module with the signal manager.
        """
def create_module(name: str) -> typing.Any:
    """
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
    """
def hadronization_hadrons_numpy(task: JetScapeTask) -> dict:
    """
    Output hadrons of a HadronizationManager (all its Hadronization tasks) or of one Hadronization task, as a dict of arrays. Call between ExecPerEvent() and ClearPerEvent().
    """
def hadronization_input_numpy(module: JetScapeTask) -> dict:
    """
    iColoredHadronization with <store_input> 1: the parton list the last event handed to Pythia (final partons, beam remnants, the colour-balancing parton), as a dict of pid, col, acol (int32), p (N, 4) float64 [E, px, py, pz] and mass float64. Empty without <store_input>. Accepts the module, its parent or the HadronizationManager.
    """
def hadronize_input(module: JetScapeTask, pid: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], col: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], acol: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], p: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mass: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], seed: typing.Any = None) -> dict:
    """
    Hadronize a parton list stored with hadronization_input_numpy again with iColoredHadronization (create_module('iColoredHadronization') after a JetScape Init() that loaded the job's XML, then module.Init()). With the stored `seed` and the same Pythia version and <JetHadronization> settings the hadrons are those of the job. Returns the hadron dict plus the seed used.
    """
def hadronize_partons(module: JetScapeTask, partons: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], seed: typing.Any = None) -> dict:
    """
    Hadronize stored final partons (an (N, 14) array, FINAL_PARTON_COLUMNS) with a jet hadronization module, e.g. create_module('ColorlessHadronization') after a JetScape Init() that loaded the XML and after module.Init(). With `seed`, the ColorlessHadronization generators are reseeded with it first (exact replay of an event hadronized with <reseed_per_event> 1). iColoredHadronization (ICOLORED_HAS_COLOUR_CLOSURE) takes the partons in their stored order, without beam remnants, through its pstat policy and <colour_closure> (jet_hadronization_colour_audit afterwards). Returns the hadron dict plus n_partons_out and the seed used.
    """
def hard_partons_numpy(task: JetScapeTask) -> numpy.typing.NDArray[numpy.float64]:
    """
            The partons the hard process handed to the framework this event (for
            PythiaGun: status 62 after ISR/MPI, or the final partons with FSR_on),
            as an (n, 7) array pid, px, py, pz, E, pT, y, hardest first.  For
            PythiaGun this is what <partonYMax> cuts on.  Call between
            ExecPerEvent() and ClearPerEvent().
    """
def hard_process_info(task: JetScapeTask) -> dict:
    """
            The current event's hard process: ``pthat`` [GeV], ``sigma_gen`` and
            ``sigma_err`` (the generator's running cross-section estimate, mb; for
            PythiaGun with several pTHat windows, the current event's window) and
            ``event_weight``.  Call between ExecPerEvent() and ClearPerEvent().
    """
def initial_state_info(task: JetScapeTask) -> dict:
    """
            The current event's geometry from its initial-state module: ``n_coll`` (binary
            collisions), ``n_part`` and ``centrality``; -1 where the module does not provide it
            (3dMCGlauber gives n_coll only).  A hard process scales with n_coll, so events of a
            wide centrality class sampled minimum-bias-like (P(b) ~ b, one hard process each)
            are weighted by it to match data.  Call between ExecPerEvent() and ClearPerEvent().
    """
def isr_hard_momentum(task: JetScapeTask) -> dict:
    """
            With initial-state radiation (PythiaIsrGun + iMATTER + 3dMCGlauber): this
            event's hard-parton four-momentum (E, px, py, pz) [GeV] handed to the initial
            state for subtraction from the projectile (``proj``, +z) and target (``targ``,
            -z) nucleon of the hard collision; ``subtract`` (1: subtracted, 0: 3dMCGlauber
            zeroes it first, -1: no such setting); and the four-momenta of those two
            nucleons' remnants (``remnant_proj``, ``remnant_targ``), which
            iColoredHadronization hadronizes.  Empty lists from a module without them.
            Call between ExecPerEvent() and ClearPerEvent().
    """
def jet_hadronization_colour_audit(module: JetScapeTask) -> dict:
    """
    iColoredHadronization: the colour audit of the last event (PLAN_lbt_colour_tags.md, C3): partons and remnants handed to Pythia, holes (pstat -1, dropped) and their energy, other negative pstat skipped, unpaired colour / anticolour tags, duplicate tags and how many were relabelled. Accepts the module, its parent or the HadronizationManager.
    """
def jet_hadronization_colour_totals(module: JetScapeTask) -> dict:
    """
    iColoredHadronization: the colour audits added up over the run (events hadronized, failed = thrown, events with open lines, holes and their energy, open ends, joins, partners, soft end partons, retries, duplicates); what the run's log summary prints.
    """
def jet_hadronization_last_random_seed(module: JetScapeTask) -> int:
    """
    The seed ColorlessHadronization (or iColoredHadronization) used for the last event (0 if it was not reseeded). Accepts the module itself, its Hadronization parent or the HadronizationManager.
    """
def jet_hadronization_set_colour_closure(module: JetScapeTask, mode: str) -> str:
    """
    iColoredHadronization: set the closure of open colour lines (reconnect, reconnect_global, thermal, fake, none) instead of the XML's <colour_closure>; returns it.
    """
def load_xml(main_xml: str, user_xml: str = '', init_random: bool = True) -> bool:
    """
              Open the main (and optionally user) XML into the JetScapeXML singleton now.
    
              Call this BEFORE constructing any C++ module that reads XML in its constructor --
              CausalLiquefier is the one that matters.  JetScape.Init() does the same thing, but
              that is too late for modules built beforehand.  Calling it twice is harmless.
    
              With init_random=True (the default) it also seeds JetScapeTaskSupport from
              <Random><seed>, so code that draws random numbers before JetScape.Init() -- such as
              InitialState.sample_binary_collision_point() -- works.
    
              Returns True if the main file parsed.
    """
def next_task_number() -> int:
    """
    The task ID the next task made will get (JetScapeTaskSupport).
    """
def pythia_gun_bins(task: JetScapeTask) -> dict:
    """
            PythiaGun's or PythiaIsrGun's pTHat windows (<Hard><PythiaGun><pTHatBins>; one window,
            pTHatMin-pTHatMax, without it): ``n_bins``, ``bins`` [(min, max)],
            ``active`` (the current event's window: event i uses i mod n_bins), and
            per window ``seeds`` (Pythia's Random:seed), ``sigma_gen``, ``sigma_err``
            [mb] and ``n_accepted`` so far.  After Init(): the windows and seeds; after
            the last event: the final cross sections.  With PYTHIA_GUN_HAS_PARTON_Y_CUT
            also ``y_max`` (0 = no cut), ``y_mode`` and per window ``n_tried`` /
            ``n_kept`` (events reaching / passing the cut), ``acceptance`` and
            ``sigma_gen_raw`` / ``sigma_err_raw`` (Pythia's own, which counts the rejected
            events; ``sigma_gen`` is it times the acceptance).  With
            PYTHIA_GUN_HAS_Z_TRIGGER also ``min_partons``, ``trigger`` ('none' or 'Z') and
            the Z acceptance ``trigger_y_max``, ``lepton_pt_min``, ``lepton_eta_max`` (0 =
            no cut); with a trigger the per-window counts and the acceptance are those of
            both cuts.
    """
def pythia_gun_trigger(task: JetScapeTask) -> dict:
    """
            PythiaGun's ``trigger`` ('none' or 'Z') and the current event's trigger
            ``leptons``: the Z's opposite-charge, same-flavour pair as handed to the
            framework (the pair with the largest invariant mass if there are several), an
            (n, 5) array pid, px, py, pz, E, hardest first; n = 2 with trigger Z, else 0.
            Call between ExecPerEvent() and ClearPerEvent().
    """
def random_seed() -> int:
    """
    The run's <Random><seed> as JetScapeTaskSupport read it (0 before JetScape.Init() or load_xml(); a time-based seed when the XML says 0).
    """
def register_module(name: str, factory: typing.Any, section: str = '', defaults: str = '') -> None:
    """
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
    """
def registered_modules(python_only: bool = False) -> list[str]:
    """
    The names in JETSCAPE's module factory (C++ modules of this build and the Python ones registered); python_only=True lists only the latter.
    """
def set_next_task_number(n: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Set the task ID the next task gets: back to next_task_number()'s value before modules made outside the physics, so later tasks keep their random seeds.
    """
def set_writer_output_file(writer: JetScapeModuleBase, file_name: str) -> None:
    """
              Set the output file of a writer created with create_module(), e.g.
              create_module("JetScapeWriterFinalStateHadronsAscii").  Call before
              JetScape.Init(), which opens the file.
    """
def soft_hadrons_numpy(task: JetScapeTask) -> dict:
    """
    All hadrons a SoftParticlization module (iSS) produced this event: dict of pid, pstat, p [E,px,py,pz], x [t,x,y,z], mass, and sample_counts (one entry per oversample; the rows are the samples concatenated in order). Call between ExecPerEvent() and ClearPerEvent().
    """
def soft_last_random_seed(task: JetScapeTask) -> int:
    """
    The seed the SoftParticlization module used for the last event.
    """
def soft_primaries_numpy(task: JetScapeTask) -> typing.Any:
    """
    The undecayed (primary) hadrons of the first N oversamples this event, as sampled before any decay (iSS <keep_primaries> N, with soft_set_compact_output): the soft_hadrons_numpy dict, sample_counts with N entries. None if none were kept. Takes them over without copying: call it once per event, after soft_hadrons_numpy.
    """
def soft_sample_cells(task: JetScapeTask, cells: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], seed: typing.SupportsInt | typing.SupportsIndex, n_samples: typing.SupportsInt | typing.SupportsIndex) -> dict:
    """
    Sample the given surface cells now, outside the framework's event loop: an (N, 32) float32 array (SURFACE_CELL_COLUMNS, e.g. a particlize file's surface unit), this seed and n_samples oversamples. Returns soft_hadrons_numpy's dict (one entry of sample_counts per oversample; none for no cells); with iSS <keep_primaries> the undecayed hadrons follow from soft_primaries_numpy. Equal to replaying the cells through the event loop with soft_set_next_random_seed. The module must be initialised (Init); it need not be in a task list.
    """
def soft_set_compact_output(task: JetScapeTask, on: bool = True) -> None:
    """
    From the next event on, hand the hadrons over as flat arrays instead of one Hadron object each: the same soft_hadrons_numpy result with ~8x less memory. soft_hadrons_numpy then takes the arrays over without copying, so call it once per event. The framework's Hadron_list_ stays empty (no bulk hadrons for writers or afterburners). iSS: ExecuteTask only.
    """
def soft_set_next_random_seed(task: JetScapeTask, seed: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Use `seed` for the next event's sampling (one-shot) instead of a draw from the module's generator.
    """
def soft_set_number_of_samples(task: JetScapeTask, n: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Samples (oversamples) per event from the next event on (iSS: number_of_repeated_sampling). Call after Init().
    """
def unregister_module(name: str) -> bool:
    """
    Remove a Python module registered with register_module(); False if the name was not one.
    """
def xml_add_user_blocks(text: str) -> int:
    """
              Append the children of ``<jetscape>`` in ``text`` to the open user XML, in memory.
    
              After JetScape.Init(), so the automatic task list never sees them: modules made
              afterwards (create_module + Init) read their settings from them, while the event
              loop keeps exactly the tasks the file had.  A block the user XML has already is
              refused.  Returns the number of blocks added.
    """
ERROR: HydroStatus
EVOLVING: HydroStatus
FINAL_PARTON_COLUMNS: tuple = ('shower', 'pid', 'pstat', 'E', 'px', 'py', 'pz', 't', 'x', 'y', 'z', 'mass', 'col', 'acol')
FINISHED: HydroStatus
HAS_GZIP_READER: bool = True
HAS_NEXT_TASK_NUMBER: bool = True
HAS_ROOT: bool = True
ICOLORED_HAS_COLOUR_AUDIT: bool = True
ICOLORED_HAS_COLOUR_CLOSURE: bool = True
ICOLORED_HAS_INPUT_RECORD: bool = True
INITIALIZED: HydroStatus
ISS_DECAYS_SMASH_LIST: bool = True
NOT_START: HydroStatus
PYTHIA_GUN_HAS_PARTON_Y_CUT: bool = True
PYTHIA_GUN_HAS_PTHAT_BINS: bool = True
PYTHIA_GUN_HAS_Z_TRIGGER: bool = True
PYTHIA_ISR_GUN_HAS_PARTON_Y_CUT: bool = True
PYTHIA_ISR_GUN_HAS_PTHAT_BINS: bool = True
PYTHIA_VERSION: int = 8312
SOFT_HAS_PRIMARIES: bool = True
SOFT_HAS_SAMPLE_CELLS: bool = True
SURFACE_CELL_COLUMNS: tuple = ('tau', 'x', 'y', 'eta', 'ds0', 'ds1', 'ds2', 'ds3', 'u0', 'u1', 'u2', 'u3', 'e', 'T', 'P', 'nB', 'nQ', 'nS', 'muB', 'muQ', 'muS', 'pi00', 'pi01', 'pi02', 'pi03', 'pi11', 'pi12', 'pi13', 'pi22', 'pi23', 'pi33', 'Pi')
