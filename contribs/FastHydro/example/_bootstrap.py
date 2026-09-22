"""Put PyJetscape and FastHydro on sys.path without an install step.

Mirrors run_music_leg.py's _import_jetscape: try the checkout layout first, then let a real
installation win if one is present.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_contribs = os.path.dirname(os.path.dirname(_here))          # .../contribs

for _p in (os.path.join(_contribs, "FastHydro", "python"),
           os.path.join(_contribs, "PyJetscape", "python")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import jetscape  # noqa: F401
except ImportError as exc:                                    # pragma: no cover
    raise SystemExit(
        f"cannot import jetscape ({exc}).\n"
        "Build it with:  cmake --build $XSCAPE_BUILD --target pyjetscape_core\n"
        "and use the interpreter it was built against (the shipped extension is cpython-313, "
        "while conda_install/install_js_fno_minimal.sh pins 3.11)."
    ) from exc
