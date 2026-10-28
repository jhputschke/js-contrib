#!/usr/bin/env python3
"""
utils/gen_pyjetscape_stubs.py

Type stubs for PyJetscape (docs/Plans/PLAN_pyjetscape_pythonic.md, step 6): run
pybind11-stubgen on the built pyjetscape_core, which also sees the Pythonic layer
(importing the module installs it), then make the layer's properties typed @property
stubs, and write contribs/PyJetscape/python/jetscape/pyjetscape_core.pyi next to the
extension, with py.typed.  Run it after a build that changes the bindings:

    conda activate js_fno
    pip install pybind11-stubgen       # once
    python utils/gen_pyjetscape_stubs.py

The stubs follow the build they are made from: names behind a build flag (ROOT, the
iColored colour functions, iSS primaries) are in them only if that build has them.
"""

import inspect
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "contribs" / "PyJetscape" / "python"
OUT = PYTHON_DIR / "jetscape" / "pyjetscape_core.pyi"

EXTRA_IMPORTS = ("import numpy as np\n"
                 "from typing import Callable, Iterator\n")


def _annotation(fget) -> str:
    ann = inspect.signature(fget).return_annotation
    if ann is inspect.Signature.empty:
        return "typing.Any"
    text = ann if isinstance(ann, str) else inspect.formatannotation(ann)
    return text.replace("core.", "")


def _properties():
    """{class name: {property name: return annotation}} of the installed layer."""
    sys.path.insert(0, str(PYTHON_DIR))
    from jetscape import pythonic

    props = {}
    for (target, name), value in pythonic._INSTALLED.items():
        if isinstance(value, pythonic._Prop):
            props.setdefault(target.__name__, {})[name] = _annotation(value.fget)
    return props


def _fix_properties(text: str, props: dict) -> str:
    blocks = re.split(r"(?m)^(?=class \w+)", text)
    out = []
    for block in blocks:
        m = re.match(r"class (\w+)", block)
        for name, ann in (props.get(m.group(1), {}) if m else {}).items():
            pattern = rf"(?m)^(?:    @staticmethod\n)?    def {name}\(\*args, \*\*kwargs\):"
            block, n = re.subn(pattern, f"    @property\n    def {name}(self) -> {ann}:",
                               block)
            if n != 1:
                raise SystemExit(f"gen_pyjetscape_stubs: {m.group(1)}.{name}: expected "
                                 f"one stub, found {n}")
        out.append(block)
    return "".join(out)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, PYTHONPATH=str(PYTHON_DIR))
        res = subprocess.run([sys.executable, "-m", "pybind11_stubgen",
                              "jetscape.pyjetscape_core", "-o", tmp,
                              "--ignore-all-errors"], env=env, capture_output=True,
                             text=True, check=False)
        stub = Path(tmp) / "jetscape" / "pyjetscape_core.pyi"
        if res.returncode != 0 or not stub.exists():
            sys.stderr.write(res.stdout + res.stderr)
            return 1
        text = stub.read_text()
    text = text.replace("from __future__ import annotations\n",
                        "from __future__ import annotations\n" + EXTRA_IMPORTS, 1)
    text = re.sub(r"\bcore\.", "", text)
    text = _fix_properties(text, _properties())
    # C++ types stubgen cannot name come out as "..." in annotations
    text = re.sub(r"(?<=-> )\.\.\.(?=:)", "typing.Any", text)
    text = re.sub(r"(?<=: )\.\.\.(?=[,)=])", "typing.Any", text)
    # the layer's FourVector.__hash__ = None (equal by value, so not hashable)
    text = text.replace("    __hash__: typing.ClassVar[None] = None\n",
                        "    __hash__: typing.ClassVar[None] = None  # type: ignore[assignment]\n")
    OUT.write_text(text)
    (OUT.parent / "py.typed").touch()
    print(f"wrote {OUT.relative_to(ROOT)} ({text.count(chr(10))} lines) and py.typed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
