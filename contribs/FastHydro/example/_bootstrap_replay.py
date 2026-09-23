"""Path shim for replay: FastHydro only -- PyJetscape is deliberately NOT required."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_p = os.path.join(os.path.dirname(_here), "python")
if os.path.isdir(_p) and _p not in sys.path:
    sys.path.insert(0, _p)
