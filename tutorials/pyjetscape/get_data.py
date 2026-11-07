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
Interrupted downloads resume (*.part); complete files are skipped.  A cache that fails
(an HTTP 5xx, a dropped connection) is passed over for the next one the director lists,
as pelicanfs does.

The default destination is $JS_TUTORIAL_DATA, else data/ next to this script.
Standard library only (Python 3.8+), so it runs in the container, with the Mac wheel, in
the analysis env and in Colab alike.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import http.client
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = ".js_transfer_crc32c.json"
USER_AGENT = "js-contrib-tutorials/get_data"
CHUNK = 1 << 20
ROUNDS = 3                     # times through the director's list of caches
RETRY_WAIT = 3                 # seconds before asking the director again, doubled after each
RETRIES = 5                    # attempts per file whose connection drops mid-download


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


# ── the caches ──────────────────────────────────────────────────────────────────────────
# The OSDF director answers a request with a redirect to the nearest cache, and lists the
# other caches that serve the object in its Link header, by priority.  Like pelicanfs, the
# requests here go to those caches in turn: a cache that fails (an HTTP 5xx, a refused,
# timed-out or dropped connection) is passed over, and tried after the others for the rest
# of the process, the more it failed the later.  When every cache has failed, the director is asked again after a pause.

_FAILED: dict = {}             # cache host -> times it failed in this process


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None            # -> HTTPError with the redirect's headers


_DIRECTOR = urllib.request.build_opener(_NoRedirect)


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc


def _failed(url: str) -> None:
    _FAILED[_host(url)] = _FAILED.get(_host(url), 0) + 1


def _name(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def _retryable(e: Exception) -> bool:
    """A failure another cache, or the same one later, may not have."""
    if isinstance(e, urllib.error.HTTPError):
        return e.code >= 500 or e.code == 429
    return isinstance(e, (urllib.error.URLError, http.client.HTTPException, ConnectionError,
                          TimeoutError, socket.timeout))


def caches(url: str, timeout: float = 30) -> list:
    """The URLs to try for ``url``: the caches the director lists for it, in its order,
    those that failed in this process after the others.  A URL that is not redirected is
    its own list."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with _DIRECTOR.open(req, timeout=timeout):
            return [url]                             # not a director
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise
        hdr = e.headers
        e.close()
    found = []
    for item in (hdr.get("Link") or "").split(","):
        m = re.match(r"\s*<([^>]+)>(.*)", item)
        if m and 'rel="duplicate"' in m.group(2):
            pri = re.search(r"pri=(\d+)", m.group(2))
            found.append((int(pri.group(1)) if pri else 1 << 30, m.group(1)))
    urls = [u for _, u in sorted(found)]
    loc = hdr.get("Location")
    if loc and loc not in urls:
        urls.insert(0, urllib.parse.urljoin(url, loc))
    return sorted(urls or [url], key=lambda u: _FAILED.get(_host(u), 0))


def _open(url: str, headers: dict | None = None, timeout: float = 60):
    """urlopen ``url`` through the OSDF caches that serve it (see above).  A 404 or 403
    fails at once; other failures move on to the next cache."""
    err = None
    for rnd in range(ROUNDS):
        if rnd:
            wait = RETRY_WAIT * 2 ** (rnd - 1)
            print(f"  {_name(url)}: no cache answered ({err}); asking the director again in "
                  f"{wait} s", file=sys.stderr)
            time.sleep(wait)
        try:
            candidates = caches(url)
        except Exception as e:                       # noqa: BLE001
            if not _retryable(e) or rnd == ROUNDS - 1:
                raise
            err = e
            continue
        for c in candidates:
            req = urllib.request.Request(c, headers={"User-Agent": USER_AGENT,
                                                     **(headers or {})})
            try:
                return urllib.request.urlopen(req, timeout=timeout)
            except Exception as e:                   # noqa: BLE001
                if not _retryable(e):
                    raise
                err = e
                _failed(c)
                print(f"  {_name(url)}: {_host(c)} failed ({e}); trying the next cache",
                      file=sys.stderr)
    raise err


def cache_url(url: str) -> str:
    """The URL of a cache that serves ``url`` now (for streaming: every read then goes to
    that cache, not through the director)."""
    with _open(url, {"Range": "bytes=0-0"}) as r:
        return r.url


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
    t0, start = time.time(), None
    for attempt in range(RETRIES):        # a connection dropped mid-file resumes from .part,
        have = part.stat().st_size if part.exists() else 0    # from the next cache
        if have > size:
            part.unlink()
            have = 0
        start = have if start is None else start
        headers = {"Range": f"bytes={have}-"} if have else {}
        r = _open(url, headers)           # passes over caches that fail to answer
        try:
            with r:
                if have and r.status != 206:      # the server ignored the range: start over
                    have = 0
                with open(part, "ab" if have else "wb") as out:
                    done = have
                    for block in iter(lambda: r.read(CHUNK), b""):
                        out.write(block)
                        done += len(block)
                        if not quiet and sys.stdout.isatty():
                            print(f"\r  ...     {dest.name}  {_fmt(done)} / {_fmt(size)}",
                                  end="")
            if part.stat().st_size >= size:
                break
            err = IOError(f"connection closed at {part.stat().st_size} of {size} bytes")
        except (http.client.HTTPException, ConnectionError, TimeoutError, socket.timeout) as e:
            err = e
        _failed(r.url)
        if attempt < RETRIES - 1:
            print(f"  {dest.name}: {_host(r.url)}: {err}; resuming from the next cache",
                  file=sys.stderr)
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
        print(f"\r  fetched {dest.name}  ({_fmt(size)}, {_fmt((size - start) / dt)}/s)" + " " * 10)
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
