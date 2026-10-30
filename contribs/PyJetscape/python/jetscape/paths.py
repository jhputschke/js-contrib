"""
python/jetscape/paths.py

Where the X-SCAPE tree, its build and the main XML are, for scripts and notebooks that live
anywhere -- a copy in a container's /work, a user's own directory -- instead of inside the
tree (docs/Plans/PLAN_container_user.md, step 1).  Pure Python, no build needed.
Paths are absolute but not resolved (symlinks kept), as the production scripts make them.

    XSCAPE          the X-SCAPE tree (the production image sets it to /opt/X-SCAPE);
                    default: the checkout this package sits in (X-SCAPE/external_packages/
                    js-contrib/contribs/PyJetscape/python/jetscape), else the installed
                    tree of the xscape wheel (macOS: site-packages/xscape/share/xscape)
    JETSCAPE_BUILD  the build tree to run; default: <XSCAPE>/build_gpu, or the installed
                    tree itself, which holds music_input, EOS and the tables

The installed tree of the wheel (``cmake --install`` of X-SCAPE) has config/ and the data of
a build tree side by side, so it serves as both XSCAPE and the build.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

__all__ = ["pyjetscape_dir", "examples_dir", "xscape_root", "build_dir", "main_xml",
           "installed_package", "installed_tree"]


def pyjetscape_dir() -> Path:
    """contribs/PyJetscape (this package is its python/jetscape) -- in a checkout."""
    return Path(os.path.abspath(__file__)).parents[2]


def installed_package() -> Path | None:
    """The xscape wheel's package directory (site-packages/xscape: bin, lib, share, the
    compiled core), or None when that wheel is not installed.  Found without importing it."""
    try:
        spec = importlib.util.find_spec("xscape")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(os.path.abspath(list(spec.submodule_search_locations)[0]))


def installed_tree() -> Path | None:
    """The xscape wheel's X-SCAPE tree (<package>/share/xscape), or None."""
    pkg = installed_package()
    tree = pkg / "share" / "xscape" if pkg is not None else None
    return tree if tree is not None and tree.is_dir() else None


def _checkout_root() -> Path | None:
    """The X-SCAPE checkout this package sits in, or None (an installed package)."""
    root = pyjetscape_dir().parents[3]
    return root if (root / "config" / "jetscape_main.xml").is_file() else None


def examples_dir() -> Path:
    """contribs/PyJetscape/example: the production scripts, notebooks and their XMLs --
    the checkout's, else the copy in the xscape wheel's installed tree."""
    local = pyjetscape_dir() / "example"
    if local.is_dir():
        return local
    tree = installed_tree()
    if tree is not None and (tree / "examples").is_dir():
        return tree / "examples"
    return local


def xscape_root() -> Path:
    """The X-SCAPE tree: $XSCAPE, else the checkout this package sits in, else the xscape
    wheel's installed tree."""
    env = os.environ.get("XSCAPE")
    if env:
        return Path(os.path.abspath(env))
    root = _checkout_root() or installed_tree()
    if root is None:
        raise FileNotFoundError(
            "jetscape.paths: no X-SCAPE tree found -- this package is not inside an X-SCAPE "
            "checkout and the xscape wheel is not installed; set XSCAPE (the tree) and, for "
            "a build other than <XSCAPE>/build_gpu, JETSCAPE_BUILD")
    return root


def build_dir() -> Path:
    """The build tree: $JETSCAPE_BUILD, else <X-SCAPE>/build_gpu, else -- an installed tree,
    which has music_input but no build_gpu -- the X-SCAPE tree itself."""
    env = os.environ.get("JETSCAPE_BUILD")
    if env:
        return Path(os.path.abspath(env))
    root = xscape_root()
    build = root / "build_gpu"
    if not build.exists() and (root / "music_input").is_file():
        return root
    return build


def main_xml() -> Path:
    """X-SCAPE's config/jetscape_main.xml."""
    return xscape_root() / "config" / "jetscape_main.xml"
