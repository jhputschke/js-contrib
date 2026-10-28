"""
python/jetscape/pythonic.py

The Pythonic layer over the bound classes (docs/Plans/PLAN_pyjetscape_pythonic.md).
Additive: install() puts new methods and properties onto the classes of pyjetscape_core
and refuses any name a class already has, so no bound method can be shadowed and every
existing script keeps working.  jetscape/__init__ calls install() once; importing
jetscape.pyjetscape_core directly runs jetscape/__init__ first, so the layer is always
there.

Naming: a method ending in _array() returns numpy data, filled once per event; a plain
noun (showers, nodes, final_partons()) returns objects.  Momentum (E, px, py, pz),
position (t, x, y, z): the order of FourVector.comp(i).

Properties on the bound classes are non-data descriptors (_Prop): an instance attribute
of the same name, as a Python module may set (self.rng = ...), still wins.
"""

from __future__ import annotations

import weakref
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from . import pyjetscape_core as core
from .hadron_array import HadronArray
from .records import (edge_records, final_parton_records, final_parton_rows,
                      hard_parton_records, initiator_records, node_records)

__all__ = ["install", "make_parton", "make_hadron", "make_photon", "module",
           "per_event", "ShowerNode"]

# dunders the layer may add; anything else with leading and trailing __ is refused
_ALLOWED_DUNDERS = frozenset({"__iter__", "__getitem__", "__array__", "__eq__", "__ne__",
                              "__hash__", "__reduce__", "__enter__", "__exit__"})
_INSTALLED: dict[tuple[Any, str], Any] = {}
_MISSING = object()
_DONE = False


class _Prop:
    """A read-only property that an instance attribute of the same name overrides
    (a non-data descriptor: no __set__)."""

    def __init__(self, fget, doc=None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.fget(obj)


def _add(target, name: str, value) -> None:
    """Install value as target.name, refusing a name target already has."""
    if _INSTALLED.get((target, name), _MISSING) is value:
        return
    is_dunder = name.startswith("__") and name.endswith("__")
    if is_dunder:
        if name not in _ALLOWED_DUNDERS:
            raise AttributeError(f"pythonic: {name} is not a dunder the layer may add")
        mro = getattr(target, "__mro__", ())
        for base in mro:
            if base is object or base.__name__ == "pybind11_object":
                continue
            if name in vars(base):
                raise AttributeError(f"pythonic: {target.__name__}.{name} exists "
                                     f"(on {base.__name__}); the layer does not replace it")
    elif hasattr(target, name):
        tname = getattr(target, "__name__", str(target))
        raise AttributeError(f"pythonic: {tname}.{name} exists; the layer only adds names")
    if isinstance(value, _Prop):
        value.name = name
    setattr(target, name, value)
    _INSTALLED[(target, name)] = value


def installed_names() -> list[str]:
    """'Class.name' for everything the layer installed (for docs and tests)."""
    return sorted(f"{getattr(t, '__name__', t)}.{n}" for t, n in _INSTALLED)


# ───────────────────────────────────────────────────────────── 4-vectors, particles
def _four(v, what: str) -> "core.FourVector":
    """A FourVector from a FourVector or a 4-sequence in comp() order (t, x, y, z) =
    (E, px, py, pz)."""
    if isinstance(v, core.FourVector):
        return core.FourVector(v)
    t, x, y, z = (float(c) for c in v)
    return core.FourVector(x, y, z, t)


def _fv_iter(self) -> Iterator[float]:
    """(t, x, y, z) = (E, px, py, pz), the order of comp(i)."""
    return iter((self.t(), self.x(), self.y(), self.z()))


def _fv_getitem(self, i: int) -> float:
    if not isinstance(i, (int, np.integer)):
        raise TypeError("FourVector index must be an integer (0 = t, 1 = x, 2 = y, 3 = z)")
    if i < 0:
        i += 4
    if not 0 <= i < 4:
        raise IndexError("FourVector index out of range")
    return self.comp(int(i))


def _fv_array(self, dtype=None, copy=None) -> np.ndarray:
    return np.array([self.t(), self.x(), self.y(), self.z()], dtype=dtype)


def _fv_eq(self, other) -> bool:
    if not isinstance(other, core.FourVector):
        return NotImplemented
    return (self.t(), self.x(), self.y(), self.z()) == (other.t(), other.x(), other.y(),
                                                         other.z())


def _fv_ne(self, other) -> bool:
    r = _fv_eq(self, other)
    return r if r is NotImplemented else not r


def _fv_reduce(self) -> tuple:
    return (core.FourVector, (self.x(), self.y(), self.z(), self.t()))


def _fv_momentum(cls, E: float, px: float, py: float, pz: float) -> "core.FourVector":
    """FourVector.momentum(E, px, py, pz) (the constructor takes px, py, pz, E)."""
    return cls(px, py, pz, E)


def _fv_position(cls, t: float, x: float, y: float, z: float) -> "core.FourVector":
    """FourVector.position(t, x, y, z) (the constructor takes x, y, z, t)."""
    return cls(x, y, z, t)


def make_parton(pid: int, momentum, position=(0.0, 0.0, 0.0, 0.0), stat: int = 0,
                label: int = 0, *, color: int = 0, anti_color: int = 0,
                cls=None) -> "core.Parton":
    """A Parton by keywords: momentum (E, px, py, pz) and position (t, x, y, z) as any
    4-sequence or a FourVector; colour tags optional.  cls=Photon for a photon."""
    cls = cls or core.Parton
    p = cls(label, pid, stat, _four(momentum, "momentum"), _four(position, "position"))
    if color:
        p.set_color(color)
    if anti_color:
        p.set_anti_color(anti_color)
    return p


def make_photon(momentum, position=(0.0, 0.0, 0.0, 0.0), stat: int = 0,
                label: int = 0) -> "core.Photon":
    """A Photon (pid 22) by keywords, as make_parton."""
    return make_parton(22, momentum, position, stat, label, cls=core.Photon)


def make_hadron(pid: int, momentum, position=(0.0, 0.0, 0.0, 0.0), stat: int = 0,
                label: int = 0, mass: float | None = None) -> "core.Hadron":
    """A Hadron by keywords: momentum (E, px, py, pz), position (t, x, y, z) as any
    4-sequence or a FourVector; without a mass the PDG mass is used (C++ Hadron)."""
    p, x = _four(momentum, "momentum"), _four(position, "position")
    if mass is None:
        return core.Hadron(label, pid, stat, p, x)
    return core.Hadron(label, pid, stat, p, x, float(mass))


def _momentum(self) -> "core.FourVector":
    """The momentum as a FourVector (a copy); iterates as (E, px, py, pz)."""
    return self.p_in()


def _position(self) -> "core.FourVector":
    """The position as a FourVector (a copy); iterates as (t, x, y, z)."""
    return self.x_in()


def _rebuild_parton(cls_name, pid, momentum, position, stat, label, color, anti_color,
                    max_color, form_time):
    p = make_parton(pid, momentum, position, stat, label, color=color, anti_color=anti_color,
                    cls=getattr(core, cls_name))
    p.set_max_color(max_color)
    p.set_form_time(form_time)
    return p


def _parton_reduce(self) -> tuple:
    return (_rebuild_parton,
            (type(self).__name__ if type(self) in (core.Parton, core.Photon) else "Parton",
             self.pid(), tuple(self.p_in()), tuple(self.x_in()), self.pstat(), self.plabel(),
             self.color(), self.anti_color(), self.max_color(), self.form_time()))


def _rebuild_hadron(pid, momentum, position, stat, label, mass, charge, baryon_number,
                    strangeness, participant):
    return core.Hadron(label, pid, stat, _four(momentum, "momentum"),
                       _four(position, "position"), mass, charge, baryon_number,
                       strangeness, participant)


def _hadron_reduce(self) -> tuple:
    return (_rebuild_hadron,
            (self.pid(), tuple(self.p_in()), tuple(self.x_in()), self.pstat(), self.plabel(),
             self.restmass(), self.charge(), self.baryon_number(), self.strangeness(),
             self.participant()))


# ───────────────────────────────────────────────────────────────────── showers
class ShowerNode:
    """A vertex of a PartonShower by id, with its position, the partons in and out and
    the split variables (PartonShower's id-based methods)."""

    __slots__ = ("shower", "id")

    def __init__(self, shower, node_id: int):
        self.shower = shower
        self.id = int(node_id)

    def __repr__(self):
        t, x, y, z = self.position
        return f"ShowerNode(id={self.id}, t={t:g}, x={x:g}, y={y:g}, z={z:g})"

    def __eq__(self, other):
        if not isinstance(other, ShowerNode):
            return NotImplemented
        return self.shower is other.shower and self.id == other.id

    def __hash__(self):
        return hash((id(self.shower), self.id))

    @property
    def vertex(self) -> "core.Vertex":
        return self.shower.vertex_by_id(self.id)

    @property
    def position(self) -> "core.FourVector":
        """FourVector (iterates as t, x, y, z)."""
        return self.vertex.x_in()

    def _edges(self, end: int) -> list:
        s = self.shower
        return [p for p in s.partons() if s.edge_of(p)[end] == self.id]

    @property
    def parents(self) -> list:
        """The partons that end here."""
        return self._edges(2)

    @property
    def children(self) -> list:
        """The partons that start here."""
        return self._edges(1)

    @property
    def split_z(self) -> float:
        return self.shower.split_z(self.id)

    @property
    def split_kt(self) -> float:
        return self.shower.split_kt(self.id)

    @property
    def split_delta_r(self) -> float:
        return self.shower.split_delta_r(self.id)

    @property
    def is_one_to_two(self) -> bool:
        return bool(self.shower.is_one_to_two(self.id))

    @property
    def is_end(self) -> bool:
        return bool(self.shower.is_end_vertex(self.id))


def _shower_iter(self) -> "Iterator[core.Parton]":
    """The partons (edges) of the shower."""
    return iter(self.partons())


def _shower_nodes(self) -> "list[ShowerNode]":
    """Every vertex as a ShowerNode, in vertices_to_numpy order."""
    ids = self.vertices_to_numpy()[:, 0]
    return [ShowerNode(self, int(i)) for i in ids]


def _shower_start_node(self, parton: "core.Parton") -> "ShowerNode":
    """The ShowerNode where `parton` starts."""
    return ShowerNode(self, self.edge_of(parton)[1])


def _shower_end_node(self, parton: "core.Parton") -> "ShowerNode":
    """The ShowerNode where `parton` ends."""
    return ShowerNode(self, self.edge_of(parton)[2])


def _shower_edge_array(self) -> np.ndarray:
    """The partons as EDGE_DTYPE records (src, tgt, pid, pstat, E px py pz, t x y z)."""
    return edge_records(self.to_numpy())


def _shower_node_array(self) -> np.ndarray:
    """The vertices as NODE_DTYPE records (id, t, x, y, z)."""
    return node_records(self.vertices_to_numpy())


# ──────────────────────────────────────────────────────────────── module classes
def _mgr_showers(self) -> "list[core.PartonShower]":
    """The PartonShowers of this event, one per shower (get_showers())."""
    return self.get_showers()


def _mgr_final_parton_array(self) -> np.ndarray:
    """The final partons as FINAL_PARTON_DTYPE records (final_partons_numpy).  After the
    manager's Exec, before ClearPerEvent."""
    return final_parton_records(self.final_partons_numpy())


def _mgr_initiator_array(self) -> np.ndarray:
    """The shower-initiating partons as INITIATOR_DTYPE records
    (showers.initiators_from_manager)."""
    from .showers import initiators_from_manager

    return initiator_records(initiators_from_manager(self))


def _soft_hadron_array(self) -> HadronArray:
    """This event's hadrons (soft_hadrons_numpy) as a HadronArray, one sample per
    oversample.  With use_compact_output() the arrays are taken over without a copy:
    call it once per event."""
    return HadronArray.from_dict(core.soft_hadrons_numpy(self))


def _soft_primary_array(self) -> "HadronArray | None":
    """The undecayed hadrons of the first N oversamples (soft_primaries_numpy) as a
    HadronArray; None if none were kept.  Once per event, after hadron_array()."""
    d = core.soft_primaries_numpy(self)
    return None if d is None else HadronArray.from_dict(d)


def _soft_use_compact_output(self, on: bool = True) -> None:
    """Hand the hadrons over as arrays from the next event on (soft_set_compact_output)."""
    core.soft_set_compact_output(self, on)


def _soft_set_number_of_samples(self, n: int) -> None:
    """Oversamples per event from the next event on (soft_set_number_of_samples)."""
    core.soft_set_number_of_samples(self, n)


def _soft_set_next_random_seed(self, seed: int) -> None:
    """Seed for the next event's sampling, one-shot (soft_set_next_random_seed)."""
    core.soft_set_next_random_seed(self, seed)


def _soft_last_random_seed(self) -> int:
    """The seed used for the last event (soft_last_random_seed)."""
    return core.soft_last_random_seed(self)


def _soft_sample_cells(self, cells, seed: int, n_samples: int) -> HadronArray:
    """Sample the given (N, 32) surface cells now (soft_sample_cells) -> HadronArray."""
    return HadronArray.from_dict(core.soft_sample_cells(self, cells, seed, n_samples))


def _hadronization_module(task):
    """task itself when it hadronizes; a manager or container: its one Hadronization
    module."""
    found = []

    def walk(t, depth):
        if depth > 2:
            return
        for sub in t.GetTaskList():
            if isinstance(sub, core.Hadronization) and not sub.GetTaskList():
                found.append(sub)
            walk(sub, depth + 1)
    if isinstance(task, core.Hadronization) and not task.GetTaskList():
        return task
    walk(task, 0)
    if len(found) != 1:
        raise ValueError(f"hadronize: {task.GetId()} holds {len(found)} hadronization "
                         "modules; call hadronize() on the module itself")
    return found[0]


def _had_hadron_array(self) -> HadronArray:
    """The hadrons this module (or every module of the manager) made this event
    (hadronization_hadrons_numpy) as a HadronArray (one sample)."""
    return HadronArray.from_dict(core.hadronization_hadrons_numpy(self))


def _had_hadronize(self, partons, seed: int | None = None) -> HadronArray:
    """Hadronize stored final partons: FINAL_PARTON_DTYPE records (final_parton_array())
    or the old (N, 14) array (hadronize_partons).  -> HadronArray; info holds the seed
    used and n_partons_out."""
    rows = final_parton_rows(partons)
    d = core.hadronize_partons(_hadronization_module(self), rows, seed)
    return HadronArray.from_dict(d)


def _had_last_random_seed(self) -> int:
    """ColorlessHadronization's (or iColoredHadronization's) last seed."""
    return core.jet_hadronization_last_random_seed(self)


def _had_colour_audit(self) -> dict:
    """iColoredHadronization: the last event's colour audit (a dict)."""
    return core.jet_hadronization_colour_audit(self)


def _had_colour_totals(self) -> dict:
    """iColoredHadronization: the audits added up over the run."""
    return core.jet_hadronization_colour_totals(self)


def _had_set_colour_closure(self, mode: str) -> None:
    """iColoredHadronization: <colour_closure> by hand (reconnect, reconnect_global,
    thermal, fake, none)."""
    core.jet_hadronization_set_colour_closure(self, mode)


def _hard_parton_array(self) -> np.ndarray:
    """The hard partons as HARD_PARTON_DTYPE records (pid, E px py pz, pt, y), hardest
    first (hard_partons_numpy)."""
    return hard_parton_records(core.hard_partons_numpy(self))


def _hard_info(self) -> dict:
    """hard_process_info(): cross section, weight, pTHat, ... of this event."""
    return core.hard_process_info(self)


def _initial_info(self) -> dict:
    """initial_state_info(): b, Npart, Ncoll, centrality, ... of this event."""
    return core.initial_state_info(self)


# ──────────────────────────────────────────────────────────────── writing modules
_PARAM_TYPES = (bool, int, float, str)


def _xml_text(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _defaults_xml(params: dict) -> str:
    for key, value in params.items():
        if not isinstance(value, _PARAM_TYPES):
            raise TypeError(f"xml_parameters[{key!r}]: int, float, str or bool, not "
                            f"{type(value).__name__}")
    return "".join(f"<{k}>{_xml_text(v)}</{k}>" for k, v in params.items())


def _read_xml_parameters(self, path_prefix: list, params: dict) -> None:
    """self.<key> = the XML value at path_prefix + [key] (user XML, else the main-XML
    default register_module put there), else the declared default."""
    for key, default in params.items():
        path = path_prefix + [key]
        try:
            text = self.get_xml_element_text(path, False)
        except RuntimeError:            # no XML loaded (a module built by hand)
            text = ""
        text = (text or "").strip()
        if not text:
            value = default
        elif isinstance(default, bool):
            value = text.lower() not in ("0", "false", "off", "no")
        elif isinstance(default, int):
            value = int(text)
        elif isinstance(default, float):
            value = float(text)
        else:
            value = text
        setattr(self, key, value)


def module(name: str, section: str = "") -> "Callable[[type], type]":
    """Class decorator: register a Python module under `name` (register_module), with a
    main-XML default built from the class's ``xml_parameters`` ({key: default}, int,
    float, str or bool).  The class's InitTask first sets self.<key> from
    [section, name, key] in the XML; the defaults are also class attributes, so a module
    built by hand has them before Init.  Use a CustomModule... name for modules the XML
    names (<Eloss><CustomModuleMySplit>)."""
    def deco(cls):
        params = dict(getattr(cls, "xml_parameters", {}) or {})
        defaults = _defaults_xml(params)
        for key, value in params.items():
            if key not in cls.__dict__:
                setattr(cls, key, value)
        prefix = [s for s in (section, name) if s]
        user_init = cls.__dict__.get("InitTask")

        def InitTask(self):
            _read_xml_parameters(self, prefix, params)
            if user_init is not None:
                return user_init(self)
            return super(cls, self).InitTask()
        InitTask.__doc__ = (user_init.__doc__ if user_init is not None else
                            f"Reads {sorted(params)} from the XML ({'/'.join(prefix)}).")
        cls.InitTask = InitTask
        cls.module_name = name
        cls.module_section = section
        core.register_module(name, cls, section=section, defaults=defaults)
        return cls
    return deco


def _module_rng(self) -> np.random.Generator:
    """A numpy Generator seeded from this task's own engine (GetMt19937Generator), made
    on first use.  Cached with the task number: a Clone() (which copies __dict__) gets a
    new task number and so its own stream.  An instance attribute `rng` overrides it.
    Python modules only: on a C++ module (no __dict__) it raises AttributeError rather
    than draw from that module's engine."""
    d = getattr(self, "__dict__", None)
    if d is None:
        raise AttributeError("rng is for modules written in Python; a C++ module draws "
                             "from its own GetMt19937Generator()")
    n = self.GetMyTaskNumber()
    cached = d.get("_jetscape_rng")
    if cached is not None and cached[0] == n:
        return cached[1]
    engine = self.GetMt19937Generator()
    seed = np.random.SeedSequence([engine() for _ in range(4)])
    gen = np.random.Generator(np.random.PCG64(seed))
    d["_jetscape_rng"] = (n, gen)
    return gen


# ───────────────────────────────────────────────────────────────── the event loop
def per_event(main_xml: str | None = None, user_xml: str | None = None,
              tasks: Sequence | None = None,
              n_events: int | None = None) -> "core.JetScapePerEvent":
    """A JetScapePerEvent driver by keywords: the XML files, a by-hand pipeline (tasks
    added in order; else the XML's task list) and the number of events.

        with js.per_event(main_xml, user_xml) as driver:
            for _ in driver.events():
                fp = driver.energy_loss_manager.final_parton_array()

    The bound constructor JetScapePerEvent() stays as it is."""
    driver = core.JetScapePerEvent()
    if main_xml is not None:
        driver.SetXMLMainFileName(str(main_xml))
    if user_xml is not None:
        driver.SetXMLUserFileName(str(user_xml))
    for t in tasks or ():
        driver.Add(t)
    if n_events is not None:            # JetScape.Init() reads <nEvents>: set it after
        _PENDING_N_EVENTS[driver] = int(n_events)
    return driver


_PENDING_N_EVENTS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _driver_enter(self) -> "core.JetScapePerEvent":
    """Init(), the number of events given to per_event(), ExecInit() (as
    run_jetscape.per_event_loop)."""
    self.Init()
    n = _PENDING_N_EVENTS.pop(self, None)
    if n is not None:
        self.SetNumberOfEvents(n)
    self.ExecInit()
    return self


def _driver_exit(self, exc_type, exc, tb) -> bool:
    """Finish()."""
    self.Finish()
    return False


def _driver_events(self, n: int | None = None,
                   on_error: "Callable[[int, Exception], None] | None" = None
                   ) -> "Iterator[int]":
    """Yield the event index once per event, after ExecPerEvent() and before
    ClearPerEvent(), so the event's results can be read; n: at most n events (default
    GetNumberOfEvents()).  Stops cleanly on break (the event is cleared); Finish() is the
    context manager's.

    on_error: when a module throws inside an event (a C++ exception, e.g. Matter's "minus
    log of sudakov negative"; the same seed hits it again on a rerun), call
    on_error(i, exc), clear the event and go on with the next one, as the productions
    skip such events.  Without it the exception propagates."""
    total = self.GetNumberOfEvents() if n is None else min(int(n), self.GetNumberOfEvents())
    for i in range(total):
        try:
            self.ExecPerEvent()
        except RuntimeError as exc:
            if on_error is None:
                raise
            on_error(i, exc)
            try:
                self.ClearPerEvent()
            except RuntimeError:
                pass
            continue
        try:
            yield i
        finally:
            self.ClearPerEvent()


def _find_task(self, cls, exclude=()):
    """The first task of type cls in the task list (two levels deep) that is none of
    `exclude`, else None."""
    def ok(t):
        return isinstance(t, cls) and not isinstance(t, exclude)
    for t in self.GetTaskList():
        if ok(t):
            return t
        for sub in t.GetTaskList():
            if ok(sub):
                return sub
    return None


def _driver_energy_loss_manager(self) -> "core.JetEnergyLossManager | None":
    """The final-state JetEnergyLossManager (the signal manager's, else the task list's;
    never the ISR chain's IsrManager inside the hard process), or None."""
    return (core.JetScapeSignalManager.Instance().GetJetEnergyLossManagerPointer()
            or _find_task(self, core.JetEnergyLossManager,
                          exclude=tuple(c for c in (getattr(core, "IsrManager", None),)
                                        if c is not None)))


def _driver_hadronization_manager(self) -> "core.HadronizationManager | None":
    """The HadronizationManager of the task list, or None."""
    return _find_task(self, core.HadronizationManager)


def _driver_soft_particlization(self) -> "core.SoftParticlization | None":
    """The (first) SoftParticlization module of the task list (e.g. iSS), or None."""
    return _find_task(self, core.SoftParticlization)


def _driver_hydro(self) -> "core.FluidDynamics | None":
    """The hydro the signal manager holds (else the first in the task list), or None."""
    return (core.JetScapeSignalManager.Instance().GetHydroPointer()
            or _find_task(self, core.FluidDynamics))


def _driver_initial_state(self) -> "core.InitialState | None":
    """The initial state the signal manager holds (else the task list's), or None."""
    return (core.JetScapeSignalManager.Instance().GetInitialStatePointer()
            or _find_task(self, core.InitialState))


# ───────────────────────────────────────────────────────────────────── install
def install() -> None:
    """Put the layer onto the bound classes and the core module (once)."""
    global _DONE
    if _DONE:
        return
    fv = core.FourVector
    _add(fv, "__iter__", _fv_iter)
    _add(fv, "__getitem__", _fv_getitem)
    _add(fv, "__array__", _fv_array)
    _add(fv, "__eq__", _fv_eq)
    _add(fv, "__ne__", _fv_ne)
    _add(fv, "__hash__", None)          # equal by value: not hashable
    _add(fv, "__reduce__", _fv_reduce)
    _add(fv, "momentum", classmethod(_fv_momentum))
    _add(fv, "position", classmethod(_fv_position))

    base = core.JetScapeParticleBase
    _add(base, "momentum", _Prop(_momentum))
    _add(base, "position", _Prop(_position))
    _add(core.Parton, "__reduce__", _parton_reduce)
    _add(core.Hadron, "__reduce__", _hadron_reduce)

    ps = core.PartonShower
    _add(ps, "__iter__", _shower_iter)
    _add(ps, "nodes", _Prop(_shower_nodes))
    _add(ps, "start_node", _shower_start_node)
    _add(ps, "end_node", _shower_end_node)
    _add(ps, "edge_array", _shower_edge_array)
    _add(ps, "node_array", _shower_node_array)

    mgr = core.JetEnergyLossManager
    _add(mgr, "showers", _Prop(_mgr_showers))
    _add(mgr, "final_parton_array", _mgr_final_parton_array)
    _add(mgr, "initiator_array", _mgr_initiator_array)

    soft = core.SoftParticlization
    _add(soft, "hadron_array", _soft_hadron_array)
    _add(soft, "primary_array", _soft_primary_array)
    _add(soft, "use_compact_output", _soft_use_compact_output)
    _add(soft, "set_number_of_samples", _soft_set_number_of_samples)
    _add(soft, "set_next_random_seed", _soft_set_next_random_seed)
    _add(soft, "last_random_seed", _Prop(_soft_last_random_seed))
    _add(soft, "sample_cells", _soft_sample_cells)

    for cls in (core.HadronizationManager, core.Hadronization):
        _add(cls, "hadron_array", _had_hadron_array)
        _add(cls, "hadronize", _had_hadronize)
        _add(cls, "last_random_seed", _Prop(_had_last_random_seed))
        if getattr(core, "ICOLORED_HAS_COLOUR_AUDIT", False):
            _add(cls, "colour_audit", _had_colour_audit)
        if getattr(core, "ICOLORED_HAS_COLOUR_CLOSURE", False):
            _add(cls, "colour_totals", _had_colour_totals)
            _add(cls, "set_colour_closure", _had_set_colour_closure)

    _add(core.HardProcess, "parton_array", _hard_parton_array)
    _add(core.HardProcess, "info", _Prop(_hard_info))
    _add(core.InitialState, "info", _Prop(_initial_info))

    _add(core.JetScapeModuleBase, "rng", _Prop(_module_rng))

    drv = core.JetScapePerEvent
    _add(drv, "__enter__", _driver_enter)
    _add(drv, "__exit__", _driver_exit)
    _add(drv, "events", _driver_events)
    _add(drv, "energy_loss_manager", _Prop(_driver_energy_loss_manager))
    _add(drv, "hadronization_manager", _Prop(_driver_hadronization_manager))
    _add(drv, "soft_particlization", _Prop(_driver_soft_particlization))
    _add(drv, "hydro", _Prop(_driver_hydro))
    _add(drv, "initial_state", _Prop(_driver_initial_state))

    from . import awkward_views

    for name, value in (("make_parton", make_parton), ("make_hadron", make_hadron),
                        ("make_photon", make_photon), ("module", module),
                        ("per_event", per_event), ("ShowerNode", ShowerNode),
                        ("HadronArray", HadronArray),
                        ("to_awkward", awkward_views.to_awkward),
                        ("select", awkward_views.select)):
        _add(core, name, value)
    _DONE = True
