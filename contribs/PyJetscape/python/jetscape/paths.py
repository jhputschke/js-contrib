"""
python/jetscape/paths.py

Where the X-SCAPE tree, its build and the main XML are, for scripts and notebooks that live
anywhere -- a copy in a container's /work, a user's own directory -- instead of inside the
tree (docs/Plans/PLAN_container_user.md, step 1).  Pure Python, no build needed.
Paths are absolute but not resolved (symlinks kept), as the production scripts make them.

    XSCAPE          the X-SCAPE tree (the production image sets it to /opt/X-SCAPE);
                    default: from where this package is, X-SCAPE/external_packages/
                    js-contrib/contribs/PyJetscape/python/jetscape
    JETSCAPE_BUILD  the build tree to run; default: <XSCAPE>/build_gpu
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["pyjetscape_dir", "examples_dir", "xscape_root", "build_dir", "main_xml"]


def pyjetscape_dir() -> Path:
    """contribs/PyJetscape (this package is its python/jetscape)."""
    return Path(os.path.abspath(__file__)).parents[2]


def examples_dir() -> Path:
    """contribs/PyJetscape/example: the production scripts, notebooks and their XMLs."""
    return pyjetscape_dir() / "example"


def xscape_root() -> Path:
    """The X-SCAPE tree: $XSCAPE, else the tree this package sits in."""
    env = os.environ.get("XSCAPE")
    return Path(os.path.abspath(env)) if env else pyjetscape_dir().parents[3]


def build_dir() -> Path:
    """The build tree: $JETSCAPE_BUILD, else <X-SCAPE>/build_gpu."""
    env = os.environ.get("JETSCAPE_BUILD")
    return Path(os.path.abspath(env)) if env else xscape_root() / "build_gpu"


def main_xml() -> Path:
    """X-SCAPE's config/jetscape_main.xml."""
    return xscape_root() / "config" / "jetscape_main.xml"
