#!/usr/bin/env python3
"""
utils/remote_transfer/js_osdf.py

Upload and download production directories and files to / from a Pelican namespace on the
OSDF (default osdf:///fno4hic), with pelicanfs.  The same tool as js_gcs.py, for Pelican:
on its first run the script makes its own virtual environment (``~/.cache/js_osdf/venv``,
plain ``python3 -m venv`` + pip) and from then on runs itself in it.  It needs
transfer_core.py next to it.

    ./js_osdf.py upload /data/AuAu_c1 --what root         # -> osdf:///fno4hic/AuAu_c1/
    ./js_osdf.py upload /data/AuAu_a /data/AuAu_b --what pair,h5 --prefix productions -j 8
    ./js_osdf.py upload /data/AuAu_c1/AuAu_c1_0003_hadrons.root   # one file
    ./js_osdf.py upload /data/out --what root --as AuAu_v1  # -> osdf:///fno4hic/AuAu_v1/
    ./js_osdf.py download AuAu_c1 --what root --to /scratch   # -> /scratch/AuAu_c1/
    ./js_osdf.py download 'AuAu_c1/*_0004_*' --to here        # files -> here/<file>
    ./js_osdf.py ls                                    # the namespace's top level
    ./js_osdf.py ls AuAu_c1 --what h5                  # files, sizes, kinds
    ./js_osdf.py setup --reinstall | --remove          # remake / delete the environment

What (``--what``, a comma list; default all) -- by file name, as for js_gcs.py:

    pair    <stem>.h5, the hydro pair file, with its <stem>.json, .xml, .log
    h5      <stem>_particlize.h5, <stem>_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5,
            <stem>_hadronize.log
    root    <stem>_hadrons.root, <campaign>_campaign.root
    all     the three, plus every other file of the directory

Where files go (--prefix, --as, --flat), single files, patterns, the skip of files already
there, incomplete HDF5 and ``*.part``: as for js_gcs.py.  Pelican gives sizes but no
checksums, so every upload also records its CRC32C in a manifest in the remote directory
(``.js_transfer_crc32c.json``) and checks the size the origin reports afterwards.  The
skip check and downloads use the manifest's CRC32C; files it doesn't know (put there by
other tools) are compared by size.  Listings, sizes and downloads come from the origin
(pelicanfs direct_reads), not from a cache that may hold an older copy.

The token (writing needs one; a public namespace reads without): --token-file, else
$JS_OSDF_TOKEN_FILE, else $BEARER_TOKEN_FILE, else $BEARER_TOKEN, else osdf.token in the
current directory, next to this script, or in ~/.config/js_osdf/; with none of them,
pelicanfs looks in the WLCG default place and can get one through the pelican CLI (OAuth).
The namespace: --namespace, else $JS_OSDF_NAMESPACE, else /fno4hic; the federation:
--federation, else osg-htc.org (the OSDF).
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transfer_core as core  # noqa: E402

ENV = core.Env("js_osdf", ("pelicanfs>=1.4", "google-crc32c>=1.5", "h5py>=3.0"),
               "JS_OSDF_ENV")
DEFAULT_NAMESPACE = "/fno4hic"
OSDF = "osg-htc.org"
TOKEN_NAME = "osdf.token"


def find_token(arg):
    """The bearer token (its text), or None: then pelicanfs looks for one itself."""
    here = Path(__file__).resolve().parent
    for c in (arg, os.environ.get("JS_OSDF_TOKEN_FILE"), os.environ.get("BEARER_TOKEN_FILE")):
        if c:                                       # given explicitly: must exist
            p = Path(c).expanduser()
            if not p.is_file():
                sys.exit(f"js_osdf: token file {c} not found")
            return p.read_text().strip()
    if os.environ.get("BEARER_TOKEN"):
        return os.environ["BEARER_TOKEN"].strip()
    for d in (Path.cwd(), here, Path("~/.config/js_osdf").expanduser()):
        if (d / TOKEN_NAME).is_file():
            return (d / TOKEN_NAME).read_text().strip()
    return None


def namespace_path(text):
    ns = "/" + text.strip().strip("/")
    if ns == "/":
        raise ValueError("the namespace can't be /")
    return ns


class OsdfStore:
    """A namespace of a Pelican federation (transfer_core's store interface)."""

    native_checksums = False
    manifest = True

    def __init__(self, namespace, federation=OSDF, token=None, fs=None):
        self.ns, self.federation = namespace, federation
        self.label = (f"osdf://{namespace}" if federation == OSDF
                      else f"pelican://{federation}{namespace}")
        self.fs = fs if fs is not None else self._make_fs(federation, token)

    @staticmethod
    def _make_fs(federation, token):
        # pelicanfs 1.4 leaves an un-awaited coroutine and unclosed sessions behind: harmless
        warnings.filterwarnings("ignore", message="coroutine .* was never awaited")
        logging.getLogger("asyncio").setLevel(logging.CRITICAL)
        from pelicanfs.core import OSDFFileSystem, PelicanFileSystem
        kw = {"direct_reads": True}                    # the origin's view, not a cache's
        if token:
            kw["headers"] = {"Authorization": f"Bearer {token}"}
        if federation == OSDF:
            return OSDFFileSystem(**kw)
        return PelicanFileSystem(f"pelican://{federation}", **kw)

    def path(self, name):
        return f"{self.ns}/{name}" if name else self.ns

    def name_of(self, path):
        p = "/" + path.lstrip("/")
        return p[len(self.ns):].strip("/") if p.startswith(self.ns) else p.strip("/")

    def relative(self, text):
        t = text.strip()
        for scheme in ("osdf://", f"pelican://{self.federation}", "pelican://"):
            if t.startswith(scheme):
                t = t[len(scheme):]
                break
        if t.startswith("/"):                          # an absolute namespace path
            if t.rstrip("/") != self.ns and not t.startswith(self.ns + "/"):
                core.die(f"{text} is not in {self.label} (use --namespace)")
            t = t[len(self.ns):]
        return t.strip("/")

    def list(self, prefix):
        try:
            found = self.fs.find(self.path(prefix.strip("/")), detail=True)
        except FileNotFoundError:
            return {}
        return {self.name_of(p): (int(info["size"]), None) for p, info in found.items()
                if info.get("type", "file") == "file"}

    def stat(self, name):
        p = self.path(name)
        self.fs.invalidate_cache(p)
        try:
            info = self.fs.info(p)
        except FileNotFoundError:
            return None
        # pelicanfs 1.4's info() says "file" for directories too; isdir() knows
        if info.get("type", "file") != "file" or self.fs.isdir(p):
            return None
        return int(info["size"]), None

    def top_level(self, prefix):
        try:
            entries = self.fs.ls(self.path(prefix.strip("/")), detail=True)
        except FileNotFoundError:
            return [], {}
        dirs = [self.name_of(e["name"]) + "/" for e in entries if e.get("type") == "directory"]
        files = {self.name_of(e["name"]): (int(e["size"]), None) for e in entries
                 if e.get("type") != "directory"}
        return sorted(dirs), files

    def upload(self, path, name):
        self.fs.put_file(str(path), self.path(name))
        self.fs.invalidate_cache(self.path(name))

    def download(self, name, path):
        self.fs.get_file(self.path(name), str(path))

    def read_bytes(self, name):
        try:
            return self.fs.cat_file(self.path(name))
        except FileNotFoundError:
            return None


def add_args(p):
    p.add_argument("--token-file", dest="token_file", default=None,
                   help=f"bearer token file (default: see above; {TOKEN_NAME})")
    p.add_argument("--namespace", type=namespace_path,
                   default=os.environ.get("JS_OSDF_NAMESPACE", DEFAULT_NAMESPACE),
                   help="namespace (default /fno4hic, or $JS_OSDF_NAMESPACE)")
    p.add_argument("--federation", default=OSDF,
                   help=f"Pelican federation (default {OSDF}, the OSDF)")


def parse_args(argv):
    return core.build_parser(__doc__, add_args, "osdf:///NAMESPACE").parse_args(argv)


def main(argv=None):
    return core.main("js_osdf", core.build_parser(__doc__, add_args, "osdf:///NAMESPACE"),
                     ENV, lambda a: OsdfStore(a.namespace, a.federation,
                                              find_token(a.token_file)),
                     ("pelicanfs", "google_crc32c"), argv)


if __name__ == "__main__":
    core.enter_env(ENV, __file__, sys.argv[1:], "JS_OSDF_NO_ENV")
    sys.exit(main())
