"""
python/jetscape/copy_examples.py

``xscape-examples [DEST]`` -- copy PyJetscape's examples into a writable directory: the
console script of the installed package (the xscape wheel), the Python twin of the
production image's utils/container/xscape-examples.

``xscape-utils [DEST]`` -- the same for js-contrib's utils/ (jobmon.py, prodwizard.py,
h5_inspect.py, remote_transfer/, ...), to read or adapt; jetscape.paths.utils_dir().

The examples write next to themselves, and an installed or image tree is read-only, so
they run from a copy:

    xscape-examples                 # -> /work/examples in the image, else ./examples
    xscape-examples ~/xscape/examples

Copies jetscape.paths.examples_dir() (production scripts, notebooks, user XMLs, configs),
without outputs (out/, out_*/, *.h5, *.root) and caches; in a git checkout, the tracked
files only.  Never overwrites: DEST must be new or empty.  The copies find the X-SCAPE
tree and its build through jetscape.paths ($XSCAPE, $JETSCAPE_BUILD, else the checkout
or the installed tree).
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys

from . import paths

#: left out of a copy made outside git: outputs and caches
_EXCLUDE = ("__pycache__", ".ipynb_checkpoints", "out", "out_*", "*.h5", "*.root")


def _in_git_checkout(path: str) -> bool:
    """A .git in ``path`` or above -- checked without running git, which on a Mac without
    the Command Line Tools opens an install dialog."""
    path = os.path.abspath(path)
    while True:
        if os.path.exists(os.path.join(path, ".git")):
            return True
        parent = os.path.dirname(path)
        if parent == path:
            return False
        path = parent


def _tracked_files(src: str) -> list[str] | None:
    """The files git tracks under ``src`` (relative paths), or None outside a checkout."""
    if not _in_git_checkout(src):
        return None
    try:
        r = subprocess.run(["git", "-C", src, "ls-files", "-z", "."], capture_output=True,
                           timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return [f for f in r.stdout.decode().split("\0") if f]


def copy_tree(src: str, dest: str, prog: str = "xscape-examples") -> str:
    """Copy ``src`` into ``dest`` (new or empty) -> ``src``: in a git checkout the tracked
    files only, else everything but outputs and caches (_EXCLUDE)."""
    if not os.path.isdir(src):
        raise FileNotFoundError(f"{prog}: nothing to copy at {src}")
    if os.path.exists(dest) and os.listdir(dest):
        raise FileExistsError(f"{prog}: {dest} exists and is not empty; give another "
                              "directory (nothing is overwritten)")
    os.makedirs(dest, exist_ok=True)
    tracked = _tracked_files(src)
    if tracked is not None:
        for rel in tracked:
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(os.path.join(src, rel), target)
    else:
        def ignore(_dir, names):
            return {n for n in names if any(fnmatch.fnmatch(n, p) for p in _EXCLUDE)}
        shutil.copytree(src, dest, ignore=ignore, dirs_exist_ok=True)
    return src


def copy_examples(dest: str) -> str:
    """Copy the examples into ``dest`` (new or empty) -> the source directory."""
    return copy_tree(str(paths.examples_dir()), dest, "xscape-examples")


def copy_utils(dest: str) -> str:
    """Copy js-contrib's utils/ into ``dest`` (new or empty) -> the source directory."""
    return copy_tree(str(paths.utils_dir()), dest, "xscape-utils")


def main(argv=None) -> int:
    default = "/work/examples" if os.path.isdir("/work") else "examples"
    p = argparse.ArgumentParser(
        prog="xscape-examples",
        description="Copy PyJetscape's examples into a writable directory.")
    p.add_argument("dest", nargs="?", default=default,
                   help=f"new or empty directory (default: {default})")
    a = p.parse_args(argv)
    dest = os.path.abspath(os.path.expanduser(a.dest))
    try:
        src = copy_examples(dest)
    except (FileNotFoundError, FileExistsError) as e:
        print(e, file=sys.stderr)
        return 1
    print(f"xscape-examples: {src} -> {dest}")
    print(f"  e.g. cd {os.path.join(dest, 'pythonic')} && python oo_workflow.py --events 1")
    return 0


def utils_main(argv=None) -> int:
    default = "/work/utils" if os.path.isdir("/work") else "utils"
    p = argparse.ArgumentParser(
        prog="xscape-utils",
        description="Copy js-contrib's utils/ (production, monitoring, transfer and "
                    "inspection tools) into a writable directory, to read or adapt.")
    p.add_argument("dest", nargs="?", default=default,
                   help=f"new or empty directory (default: {default})")
    a = p.parse_args(argv)
    dest = os.path.abspath(os.path.expanduser(a.dest))
    try:
        src = copy_utils(dest)
    except (FileNotFoundError, FileExistsError) as e:
        print(e, file=sys.stderr)
        return 1
    print(f"xscape-utils: {src} -> {dest}")
    print(f"  e.g. python {os.path.join(dest, 'h5_inspect.py')} --help")
    return 0


if __name__ == "__main__":
    sys.exit(main())
