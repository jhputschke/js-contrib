"""
python/jetscape/run_jetscape.py

High-level simulation driver helpers.

Two modes are supported — matching the two values of
<enableAutomaticTaskListDetermination> in the JETSCAPE XML:

Mode A — automatic task list  (enableAutomaticTaskListDetermination = true)
    run_automatic(main_xml, user_xml)
    All modules are instantiated from the XML by the C++ factory.  No Python
    module injection is possible in this mode.

Mode B — manual task list  (enableAutomaticTaskListDetermination = false)
    run_manual(main_xml, user_xml, modules)
    Modules are supplied explicitly as a Python list.  Python trampoline
    modules (e.g. PyFNOHydro) are supported.  The helper validates that the
    supplied list is in a legal pipeline order before calling js.Add().

Mode C — per-event external loop  (either XML configuration)
    per_event_loop(main_xml, user_xml, ...)   # generator, yields once per event
    run_per_event(main_xml, user_xml, on_event, ...)   # callback per event
    Uses JetScapePerEvent so the event loop runs in Python.  The per-event
    module results are accessible (e.g. via JetScapeSignalManager) *after* the
    event executes and *before* its memory is released.  Works with the
    automatic (modules=None) or manual (modules=[...]) task list.

Usage example (Mode C):
    from jetscape import JetScapeSignalManager
    from jetscape.run_jetscape import per_event_loop

    for js in per_event_loop("config/jetscape_main.xml",
                             "config/jetscape_user.xml"):
        hydro = JetScapeSignalManager.Instance().GetHydroPointer()
        if hydro is not None:
            info = hydro.get_bulk_info()

Usage example (Mode A):
    from python.jetscape.run_jetscape import run_automatic
    js = run_automatic("config/jetscape_main.xml", "config/jetscape_user_MUSIC.xml")

Usage example (Mode B):
    from python.jetscape import create_module
    from python.jetscape.fno_hydro import PyFNOHydro
    from python.jetscape.run_jetscape import run_manual

    ini      = create_module("TrentoInitial")
    preeq    = create_module("FreestreamMilne")
    fno      = PyFNOHydro("models/traced.pt", config)
    jloss_mgr = create_module("JetEnergyLossManager")
    jloss     = create_module("JetEnergyLoss")
    matter    = create_module("Matter")
    jloss.Add(matter)
    jloss_mgr.Add(jloss)

    js = run_manual(
        "config/jetscape_main.xml",
        "config/jetscape_user_fno_python.xml",
        [ini, preeq, fno, jloss_mgr],
    )
"""

from __future__ import annotations

from typing import Callable, Iterator, Sequence

from .pyjetscape_core import (
    JetScape,
    JetScapePerEvent,
    JetScapeTask,
    InitialState,
    PreequilibriumDynamics,
    FluidDynamics,
)

# ── Pipeline stage order for Mode B validation ────────────────────────────────
# Only the *top-level* module types are listed here.
# JetEnergyLoss algorithms (Matter, Martini, …) and Hadronization algorithms
# are nested inside their managers and should NOT appear in the top-level list.
_STAGE_ORDER: list[type] = [
    InitialState,          # e.g. TrentoInitial
    PreequilibriumDynamics, # e.g. FreestreamMilne, NullPreDynamics
    FluidDynamics,         # e.g. FnoHydro, PyFNOHydro, MusicWrapper
    # JetEnergyLossManager, HadronizationManager, SoftParticlization bindings
    # are available via create_module(); they inherit JetScapeModuleBase.
    # They are accepted by run_manual() without strict ordering checks
    # (since their pybind11 classes are not imported here to avoid circular deps).
]


def _validate_pipeline_order(modules: Sequence, stage_order: list[type]) -> None:
    """
    Check that top-level modules in *modules* appear in a valid stage order.

    Raises
    ------
    ValueError
        If a module appears before an earlier expected stage.
    """
    last_stage_idx = -1
    for mod in modules:
        for i, stage_type in enumerate(stage_order):
            if isinstance(mod, stage_type):
                if i < last_stage_idx:
                    raise ValueError(
                        f"{type(mod).__name__} is out of pipeline order. "
                        f"It must come after stage index {last_stage_idx} "
                        f"({stage_order[last_stage_idx].__name__})."
                    )
                last_stage_idx = i
                break


# ── Mode A ────────────────────────────────────────────────────────────────────

def run_automatic(main_xml: str, user_xml: str) -> JetScape:
    """
    Run a fully XML-driven JETSCAPE pipeline.

    The XML must contain::

        <enableAutomaticTaskListDetermination>true</enableAutomaticTaskListDetermination>

    All modules are instantiated from the XML by the C++ factory.  No Python
    module injection is possible in this mode.

    Parameters
    ----------
    main_xml : str
        Path to the main (default parameters) XML file.
    user_xml : str
        Path to the user (overrides) XML file.

    Returns
    -------
    JetScape
        The framework object after Finish() has been called.
    """
    js = JetScape()
    js.SetXMLMainFileName(main_xml)
    js.SetXMLUserFileName(user_xml)
    js.Init()
    js.Exec()
    js.Finish()
    return js


# ── Mode B ────────────────────────────────────────────────────────────────────

def run_manual(
    main_xml: str,
    user_xml: str,
    modules: Sequence,
    n_events: int | None = None,
) -> JetScape:
    """
    Run a JETSCAPE pipeline with an explicit module list.

    The XML must contain::

        <enableAutomaticTaskListDetermination>false</enableAutomaticTaskListDetermination>

    Modules are added in the order supplied.  Python trampoline modules
    (e.g. PyFNOHydro) are supported alongside C++ modules obtained via
    create_module().

    The helper validates that the supplied top-level modules are in a legal
    pipeline stage order before handing them to JetScape.Add().

    Parameters
    ----------
    main_xml : str
        Path to the main (default parameters) XML file.
    user_xml : str
        Path to the user (overrides) XML file.
    modules : sequence of JetScapeTask
        Top-level modules added to JetScape in pipeline order, e.g.::

            [ini, preeq, fno, jloss_manager, hadro_manager]

        Note: sub-modules (JetEnergyLoss, Hadronization, etc.) must already
        be nested inside their managers via manager.Add() before calling
        run_manual().
    n_events : int, optional
        Override the number of events from the XML.

    Returns
    -------
    JetScape
        The framework object after Finish() has been called.

    Raises
    ------
    ValueError
        If the module list is in an illegal pipeline order.
    """
    _validate_pipeline_order(modules, _STAGE_ORDER)

    js = JetScape()
    js.SetXMLMainFileName(main_xml)
    js.SetXMLUserFileName(user_xml)

    for mod in modules:
        js.Add(mod)

    if n_events is not None:
        js.SetNumberOfEvents(n_events)

    js.Init()
    js.Exec()
    js.Finish()
    return js


# ── Mode C — per-event external loop ──────────────────────────────────────────
# JetScapePerEvent exposes the event loop one event at a time, so the loop runs
# here in Python and the per-event module results can be read before their
# memory is released. Works in both automatic (modules=None) and manual
# (modules=[...]) configurations.

def _build_per_event_driver(
    main_xml: str,
    user_xml: str,
    modules: Sequence | None,
    n_events: int | None,
    start_event: int,
) -> JetScapePerEvent:
    """Construct, configure and initialise a JetScapePerEvent driver.

    Performs everything up to (and including) ExecInit(), so the returned
    object is ready for the first ExecPerEvent() call. Shared by
    per_event_loop() and run_per_event().
    """
    js = JetScapePerEvent()
    js.SetXMLMainFileName(main_xml)
    js.SetXMLUserFileName(user_xml)

    if modules:
        _validate_pipeline_order(modules, _STAGE_ORDER)
        for mod in modules:
            js.Add(mod)

    if n_events is not None:
        js.SetNumberOfEvents(n_events)

    js.Init()

    if start_event:
        js.SetStartEvent(start_event)

    js.ExecInit()
    return js


def per_event_loop(
    main_xml: str,
    user_xml: str,
    modules: Sequence | None = None,
    n_events: int | None = None,
    start_event: int = 0,
) -> Iterator[JetScapePerEvent]:
    """Iterate over a JETSCAPE run one event at a time.

    Yields the JetScapePerEvent driver once per event, *after* the event has
    been executed but *before* its memory is released, so the caller can read
    the live module results for that event::

        from jetscape import JetScapeSignalManager
        from jetscape.run_jetscape import per_event_loop

        for js in per_event_loop(main_xml, user_xml):
            hydro = JetScapeSignalManager.Instance().GetHydroPointer()
            if hydro is not None:
                info = hydro.get_bulk_info()
            # ... analyse this event ...
        # Finish() is called automatically when the loop ends.

    The memory for each event is released (ClearPerEvent) immediately after the
    consumer resumes the generator, and Finish() is called when iteration ends
    or the generator is closed (e.g. on an early `break`).

    Parameters
    ----------
    main_xml : str
        Path to the main (default parameters) XML file.
    user_xml : str
        Path to the user (overrides) XML file.
    modules : sequence of JetScapeTask, optional
        Manual (Mode B) pipeline. If omitted, the task list is taken from the
        XML (Mode A). When given, the same pipeline-order validation as
        run_manual() is applied.
    n_events : int, optional
        Override the number of events from the XML.
    start_event : int, default 0
        Starting event number for the global event counter.

    Yields
    ------
    JetScapePerEvent
        The driver object, with the current event's data available in memory.
    """
    js = _build_per_event_driver(main_xml, user_xml, modules, n_events,
                                 start_event)
    try:
        for _ in range(js.GetNumberOfEvents()):
            js.ExecPerEvent()
            yield js
            js.ClearPerEvent()
    finally:
        js.Finish()


def run_per_event(
    main_xml: str,
    user_xml: str,
    on_event: Callable[[JetScapePerEvent], None],
    modules: Sequence | None = None,
    n_events: int | None = None,
    start_event: int = 0,
) -> JetScapePerEvent:
    """Run a JETSCAPE simulation, invoking a callback once per event.

    Callback-based counterpart to per_event_loop(). ``on_event(js)`` is called
    for each event after ExecPerEvent() and before ClearPerEvent(), i.e. while
    the event's module results are still in memory::

        def analyse(js):
            hydro = JetScapeSignalManager.Instance().GetHydroPointer()
            ...

        run_per_event(main_xml, user_xml, analyse)

    Parameters
    ----------
    main_xml, user_xml : str
        XML configuration file paths.
    on_event : callable
        Called as ``on_event(js)`` for each event, with the live data in memory.
    modules : sequence of JetScapeTask, optional
        Manual (Mode B) pipeline; if omitted the XML task list (Mode A) is used.
    n_events : int, optional
        Override the number of events from the XML.
    start_event : int, default 0
        Starting event number for the global event counter.

    Returns
    -------
    JetScapePerEvent
        The driver object after Finish() has been called.
    """
    js = _build_per_event_driver(main_xml, user_xml, modules, n_events,
                                 start_event)
    try:
        for _ in range(js.GetNumberOfEvents()):
            js.ExecPerEvent()
            on_event(js)
            js.ClearPerEvent()
    finally:
        js.Finish()
    return js
