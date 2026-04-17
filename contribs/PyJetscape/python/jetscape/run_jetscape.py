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

from typing import Sequence

from .pyjetscape_core import (
    JetScape,
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
