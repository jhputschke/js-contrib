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
