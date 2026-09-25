"""The parton shower as a space-time graph.

Moved to PyJetscape as `jetscape.showers`, so the MUSIC pair writer can store showers without
importing FastHydro (which imports jetscape).  Re-exported here so existing imports keep
working.

Importing the `jetscape` *package* loads `pyjetscape_core` (and torch through `fno_hydro`),
which `import fasthydro` promises not to do.  So in the js-contrib tree the module file is
loaded directly; an installed PyJetscape, or one already imported, is used as it is.
"""

import importlib.util as _ilu
import pathlib as _pl
import sys as _sys


def _load():
    if "jetscape.showers" in _sys.modules:           # one module object per process
        return _sys.modules["jetscape.showers"]
    # contribs/FastHydro/python/fasthydro/showers.py -> contribs/PyJetscape/python/jetscape/
    src = (_pl.Path(__file__).resolve().parents[3]
           / "PyJetscape" / "python" / "jetscape" / "showers.py")
    if not src.is_file():
        import jetscape.showers as mod
        return mod
    name = "fasthydro._jetscape_showers"
    if name not in _sys.modules:
        spec = _ilu.spec_from_file_location(name, src)
        mod = _ilu.module_from_spec(spec)
        _sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return _sys.modules[name]


_mod = _load()
__all__ = list(_mod.__all__)
globals().update({k: getattr(_mod, k) for k in __all__})
