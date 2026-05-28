"""
examples/per_event_loop.py

Drive a JETSCAPE simulation one event at a time from Python, using the new
JetScapePerEvent class. The event loop runs here (not inside JetScape::Exec),
so the per-event module results can be inspected *after* each event executes
and *before* its memory is released.

This is the Python counterpart of examples/custom_examples/JetScapePerEventTest.cc.

Usage (Mode A — XML-driven task list; user XML has
       enableAutomaticTaskListDetermination = true):
    conda activate js_fno
    python examples/per_event_loop.py \\
        --main config/jetscape_main.xml \\
        --user config/jetscape_user.xml \\
        --events 5

Usage (Mode B — manual pipeline; user XML has
       enableAutomaticTaskListDetermination = false):
    python examples/per_event_loop.py --manual \\
        --user config/jetscape_user_MUSIC.xml \\
        --initial-state TrentoInitial \\
        --preequilibrium NullPreDynamics \\
        --hydro-module MUSIC \\
        --events 5

What it demonstrates:
    * jetscape.run_jetscape.per_event_loop(...) as a generator that yields the
      driver once per event, with either the XML task list (--xml, default) or
      an explicit module list (--manual).
    * Reading the live hydro module (via JetScapeSignalManager) for the current
      event before ClearPerEvent() releases it. Because the per-event memory is
      still intact at the yield point, no set_preserve_bulk_info() is needed.
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Make sure the python package is importable ────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--main",
                   default=os.path.join(_REPO_ROOT, "config", "jetscape_main.xml"),
                   help="Path to the main XML config.")
    p.add_argument("--user",
                   default=os.path.join(_REPO_ROOT, "config", "jetscape_user.xml"),
                   help="Path to the user XML config.")
    p.add_argument("--events", type=int, default=None,
                   help="Override the number of events from the XML.")
    p.add_argument("--start-event", type=int, default=0, dest="start_event",
                   help="Starting event number for the global counter (default 0).")
    # ── Manual (Mode B) pipeline options ──────────────────────────────────────
    p.add_argument("--manual", action="store_true",
                   help="Build the pipeline manually (like run_manual) instead "
                        "of using the XML task list. The user XML must have "
                        "enableAutomaticTaskListDetermination = false.")
    p.add_argument("--initial-state", default="TrentoInitial",
                   dest="initial_state",
                   help="Manual mode: initial-state module name "
                        "(default: TrentoInitial).")
    p.add_argument("--preequilibrium", default="NullPreDynamics",
                   help="Manual mode: pre-equilibrium module name "
                        "(default: NullPreDynamics).")
    p.add_argument("--hydro-module", default="MUSIC", dest="hydro_module",
                   help="Manual mode: hydro module name (default: MUSIC).")
    return p.parse_args()


def build_manual_pipeline(args) -> list:
    """Build an explicit bulk pipeline [initial-state, pre-equilibrium, hydro].

    Mirrors the manual (Mode B) module set used in inspect_bulk_info.py, kept
    minimal here since the example only needs hydro output to demonstrate the
    per-event access point. All three modules are created via the C++ factory;
    swap in Python trampoline modules (e.g. PyFNOHydro) as needed.
    """
    from jetscape import create_module

    ini   = create_module(args.initial_state)
    preeq = create_module(args.preequilibrium)
    hydro = create_module(args.hydro_module)
    return [ini, preeq, hydro]


def main() -> None:
    args = parse_args()

    from jetscape import JetScapeSignalManager
    from jetscape.run_jetscape import per_event_loop

    # Mode B (manual pipeline) when --manual, else Mode A (XML task list).
    modules = build_manual_pipeline(args) if args.manual else None
    if modules is not None:
        print("Manual pipeline: "
              + " -> ".join(m.GetId() for m in modules))

    n_seen = 0
    for js in per_event_loop(args.main, args.user,
                             modules=modules,
                             n_events=args.events,
                             start_event=args.start_event):
        # --- External access point: the current event's data is live here. ---
        # 'js' is the JetScapePerEvent driver; GetCurrentEvent() is inherited
        # (static) from JetScapeModuleBase and reflects any --start-event offset.
        event = js.GetCurrentEvent()
        sm = JetScapeSignalManager.Instance()
        hydro = sm.GetHydroPointer()   # FluidDynamics or None

        msg = f"[event {event}] "
        if hydro is not None:
            try:
                bulk = hydro.get_bulk_info()
                msg += f"hydro bulk_info available ({type(bulk).__name__})"
            except Exception as exc:  # module may not expose bulk info
                msg += f"hydro present, bulk_info unavailable ({exc})"
        else:
            msg += "no hydro module registered"
        print(msg)

        n_seen += 1
        # ClearPerEvent() runs automatically when the loop resumes.

    print(f"Done. Processed {n_seen} event(s).")


if __name__ == "__main__":
    main()
