#!/usr/bin/env python3
"""
get_data.py -- download the tutorials' data from the OSDF, with nothing but Python.

    python get_data.py --list                         # the sets, their sizes and status
    python get_data.py oo_test                        # -> data/oo_test/
    python get_data.py auau_jet_event --to ~/data     # -> ~/data/auau_jet_event/
    python get_data.py auau_hadrons_root --max-files 3
    python get_data.py auau_jet_event --dry-run       # what would be fetched
    python get_data.py auau_pair_mini --include '*_hadrons_*.h5' '*_particlize.h5' 

The sets are listed in datasets.json next to this script.  The files come over plain
HTTPS from the OSDF director, which redirects to a nearby cache: no login, no curl, no
pelicanfs, no X-SCAPE.  Each OSDF folder holds the manifest js_osdf.py writes on upload
(.js_transfer_crc32c.json), which gives the file names and sizes; a file is complete when
its size matches, and its CRC32C is checked too when google-crc32c or crc32c is installed.
Interrupted downloads resume (*.part); complete files are skipped.

The default destination is $JS_TUTORIAL_DATA, else data/ next to this script.
Standard library only (Python 3.8+), so it runs in the container, with the Mac wheel, in
the analysis env and in Colab alike.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = ".js_transfer_crc32c.json"
USER_AGENT = "js-contrib-tutorials/get_data"
CHUNK = 1 << 20


def load_sets(path: Path = HERE / "datasets.json") -> dict:
    with open(path) as f:
        return json.load(f)


def default_dest() -> Path:
    env = os.environ.get("JS_TUTORIAL_DATA")
    return Path(env).expanduser() if env else HERE / "data"


def base_url(cfg: dict) -> str:
    return cfg["osdf_https"].rstrip("/") + "/" + cfg["namespace"].strip("/")


def url_of(cfg: dict, set_name: str, filename: str = "") -> str:
    """The HTTPS URL of a file of a set (or of its folder, without ``filename``)."""
    folder = cfg["sets"][set_name]["path"].strip("/")
    return f"{base_url(cfg)}/{folder}/{filename}"


def _open(url: str, headers: dict | None = None, timeout: float = 60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def manifest(cfg: dict, set_name: str) -> dict:
    """{filename: {"size": int, "crc32c": str}} of the set's folder, filtered by 'include'."""
    with _open(url_of(cfg, set_name, MANIFEST)) as r:
        files = json.load(r)["files"]
    include = cfg["sets"][set_name].get("include")
    if include:
        files = {k: v for k, v in files.items() if any(fnmatch.fnmatch(k, p) for p in include)}
    return dict(sorted(files.items()))


def _crc32c_of(path: Path) -> str | None:
    """The base64 CRC32C (as GCS and the manifest write it), or None without a crc32c module."""
    try:
        import google_crc32c
        h = google_crc32c.Checksum()
        update, digest = h.update, h.digest
    except ImportError:
        try:
            import crc32c
            state = [0]

            def update(b):
                state[0] = crc32c.crc32c(b, state[0])

            def digest():
                return state[0].to_bytes(4, "big")
        except ImportError:
            return None
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * CHUNK), b""):
            update(block)
    return base64.b64encode(digest()).decode()


def _short(path) -> str:
    """``path`` with the home directory written as ~ (output gets shared in notebooks)."""
    s, home = str(path), str(Path.home())
    return "~" + s[len(home):] if home not in ("", "/") and s.startswith(home) else s


def _fmt(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1000
    return ""


def fetch_file(url: str, dest: Path, size: int, crc: str | None = None,
               quiet: bool = False) -> Path:
    """Download ``url`` to ``dest`` (resuming a .part), check the size and, if possible, the
    CRC32C.  A complete file already there is kept."""
    if dest.exists() and dest.stat().st_size == size:
        if not quiet:
            print(f"  ok      {dest.name}  ({_fmt(size)}, already there)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    t0 = time.time()
    headers = {"Range": f"bytes={have}-"} if have else {}
    with _open(url, headers) as r:
        if have and r.status != 206:      # the server ignored the range: start over
            have = 0
        with open(part, "ab" if have else "wb") as out:
            done = have
            for block in iter(lambda: r.read(CHUNK), b""):
                out.write(block)
                done += len(block)
                if not quiet and sys.stdout.isatty():
                    print(f"\r  ...     {dest.name}  {_fmt(done)} / {_fmt(size)}", end="")
    got = part.stat().st_size
    if got != size:
        raise IOError(f"{dest.name}: got {got} bytes, expected {size}; run again to resume")
    if crc:
        mine = _crc32c_of(part)
        if mine is not None and mine != crc:
            part.unlink()
            raise IOError(f"{dest.name}: CRC32C {mine} != {crc}; deleted, run again")
    part.replace(dest)
    if not quiet:
        dt = max(time.time() - t0, 1e-3)
        print(f"\r  fetched {dest.name}  ({_fmt(size)}, {_fmt((size - have) / dt)}/s)" + " " * 10)
    return dest


def fetch_set(set_name: str, dest_root: Path | None = None, max_files: int | None = None,
              dry_run: bool = False, cfg: dict | None = None, quiet: bool = False,
              include: list | None = None) -> Path:
    """Download a set into ``dest_root/set_name`` -> that directory.  ``include``: only the
    files matching one of these globs."""
    cfg = cfg or load_sets()
    if set_name not in cfg["sets"]:
        raise KeyError(f"no set {set_name!r}; known: {', '.join(cfg['sets'])}")
    info = cfg["sets"][set_name]
    if info.get("status") == "planned":
        raise RuntimeError(f"{set_name} is planned, not uploaded yet ({info['path']})")
    dest = (dest_root or default_dest()) / set_name
    files = manifest(cfg, set_name)
    if include:
        files = {k: v for k, v in files.items() if any(fnmatch.fnmatch(k, p) for p in include)}
    if max_files:
        files = dict(list(files.items())[:max_files])
    total = sum(v["size"] for v in files.values())
    if not quiet:
        print(f"{set_name}: {len(files)} file(s), {_fmt(total)} -> {_short(dest)}")
    for name, meta in files.items():
        if dry_run:
            print(f"  would fetch {name}  ({_fmt(meta['size'])})")
            continue
        fetch_file(url_of(cfg, set_name, name), dest / name, meta["size"],
                   meta.get("crc32c"), quiet=quiet)
    return dest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="get_data.py", description=__doc__.split("\n\n")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Sets: python get_data.py --list.  Destination: --to, $JS_TUTORIAL_DATA, "
               "else data/ next to this script.")
    p.add_argument("sets", nargs="*", help="data sets to download (datasets.json)")
    p.add_argument("--list", action="store_true", help="list the sets and exit")
    p.add_argument("--to", type=Path, help="destination directory (one folder per set)")
    p.add_argument("--max-files", type=int, help="take only the first N files of each set")
    p.add_argument("--include", nargs="+", metavar="GLOB",
                   help="only the files matching one of these patterns (quote them)")
    p.add_argument("--dry-run", action="store_true", help="show what would be fetched")
    a = p.parse_args(argv)
    cfg = load_sets()
    if a.list or not a.sets:
        print(f"Data sets (from {base_url(cfg)}):")
        for name, s in cfg["sets"].items():
            print(f"  {name:20s} {s['size']:>9s}  {s['status']:9s}  {s['description']}")
        if not a.sets:
            print("\nDownload with: python get_data.py SET [SET ...] [--to DIR]")
        return 0
    try:
        for name in a.sets:
            fetch_set(name, a.to.expanduser() if a.to else None, a.max_files, a.dry_run, cfg,
                      include=a.include)
    except (KeyError, RuntimeError, IOError, urllib.error.URLError) as e:
        print(f"get_data.py: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
