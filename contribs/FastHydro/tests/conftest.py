"""Test configuration for the FastHydro contribution.

Two families live here:

* ``test_fast_data_*.py`` -- vendored verbatim from FNO4d's own suite, with a single edit:
  the path shim points at ``../python`` (the vendored tree) instead of FNO4d's ``loc_libs``.
  They prove the solver still behaves after vendoring.  A few of them reach for FNO4d-only
  modules (``generate``, ``data.dataset``, ``read_3d_hdf5``) that this contribution does not
  ship, and skip themselves when those are absent.

* ``test_adapter_*.py`` / ``test_two_stage_*.py`` etc. -- new, covering the JETSCAPE glue.
  Anything needing a built ``pyjetscape_core`` is marked ``needs_xscape`` and skips when the
  extension is not importable, so ``pytest tests`` works on a machine with no X-SCAPE build.
"""
import os
import pathlib
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.join(_HERE, "..", "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

# PyJetscape, if this is a js-contrib checkout
_PYJS = os.path.join(_HERE, "..", "..", "PyJetscape", "python")
if os.path.isdir(_PYJS) and _PYJS not in sys.path:
    sys.path.insert(0, _PYJS)


def _have_xscape():
    try:
        import jetscape.pyjetscape_core  # noqa: F401
        return True
    except Exception:
        return False


HAVE_XSCAPE = _have_xscape()


def pytest_collection_modifyitems(config, items):
    if HAVE_XSCAPE:
        return
    skip = pytest.mark.skip(
        reason="needs a built pyjetscape_core "
               "(cmake --build $XSCAPE_BUILD --target pyjetscape_core), imported with the "
               "interpreter it was built against")
    for item in items:
        if "needs_xscape" in item.keywords:
            item.add_marker(skip)


# ── framework bootstrap ───────────────────────────────────────────────────────
# Some framework calls need state that JetScape::Init() normally sets up. The one that bites
# here is the task-support RNG: InitialState::SampleABinaryCollisionPoint draws from it and
# throws "Trying to use JetScapeTaskSupport::GetMt19937Generator before initialization"
# otherwise. load_xml(..., init_random=True) seeds it from <Random><seed>.
#
# js-contrib is cloned into X-SCAPE/external_packages/, so the main XML is five levels up.
_MAIN_XML = pathlib.Path(_HERE).resolve().parents[4] / "config" / "jetscape_main.xml"
_USER_XML = pathlib.Path(_HERE).resolve().parent / "config" / "jetscape_user_fasthydro.xml"


@pytest.fixture(scope="session", autouse=True)
def xscape_xml():
    """Load the XML singleton and seed the RNG once per session, if X-SCAPE is around."""
    if not HAVE_XSCAPE or not _MAIN_XML.exists():
        return None
    from jetscape.pyjetscape_core import load_xml
    load_xml(str(_MAIN_XML), str(_USER_XML) if _USER_XML.exists() else "")
    return str(_MAIN_XML)
