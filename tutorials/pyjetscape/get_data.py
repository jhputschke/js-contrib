#!/usr/bin/env python3
"""
get_data.py -- download the tutorials' data from the OSDF.

    python get_data.py --list                         # the sets, their sizes and status
    python get_data.py oo_test                        # -> data/oo_test/
    python get_data.py auau_jet_event --to ~/data     # -> ~/data/auau_jet_event/
    python get_data.py auau_hadrons_root --max-files 3
    python get_data.py auau_jet_event --dry-run       # what would be fetched
    python get_data.py auau_pair_mini --include '*_hadrons_*.h5' '*_particlize.h5' 

The sets are listed in datasets.json next to this script.  The files are read with
pelicanfs, the OSDF's client for Python (fsspec's osdf://), which every setup installs: it
asks the OSDF director for the caches that serve a file and reads from the nearest one, no
login needed.  Without pelicanfs (older installs), or with JS_TUTORIAL_HTTPS=1, the files
come over plain HTTPS with the standard library alone, through the same caches.

Each OSDF folder holds the manifest js_osdf.py writes on upload (.js_transfer_crc32c.json),
which gives the file names and sizes; a file is complete when its size matches, and its
CRC32C is checked too when google-crc32c or crc32c is installed.  Interrupted downloads
resume (*.part); complete files are skipped.  A cache that fails (an HTTP 5xx, a dropped
connection) is passed over for the next one.

The default destination is $JS_TUTORIAL_DATA, else data/ next to this script.
Python 3.8+; pelicanfs is used when it is installed.
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
RETRIES = 5                    # attempts per read (pelicanfs) or per dropped download
BLOCK = 16 << 20               # bytes per read when downloading with pelicanfs


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


def osdf_path(cfg: dict, set_name: str, filename: str = "") -> str:
    """The OSDF name of a file of a set, /fno4hic/...; osdf:// in front makes its URL."""
    folder = cfg["sets"][set_name]["path"].strip("/")
    return f"/{cfg['namespace'].strip('/')}/{folder}/{filename}"


# ── pelicanfs ───────────────────────────────────────────────────────────────────────────
# pelicanfs keeps the director's list of caches for a namespace; a read that fails marks
# its cache bad and raises, so the next call goes to the next cache.  _pelican() calls
# again until a read succeeds, and starts over with the director when no cache is left.

_FS = None                     # pelicanfs's file system; False when it cannot be used


def osdf_fs(fresh: bool = False):
    """pelicanfs's file system for the OSDF (fsspec's "osdf"), or None: pelicanfs not
    installed, or JS_TUTORIAL_HTTPS=1."""
    global _FS
    if _FS is None or fresh:
        _FS = False
        if not os.environ.get("JS_TUTORIAL_HTTPS"):
            try:
                import fsspec
                import pelicanfs  # noqa: F401
                if _pelican_broken():
                    return None
                _FS = fsspec.filesystem("osdf", skip_instance_cache=fresh)
            except Exception:                        # noqa: BLE001
                _FS = False
    return _FS or None


def _pelican_broken() -> bool:
    """pelicanfs up to 1.5.0 with aiowebdav2 1.x (2026-11-02 on) cannot stat or list files,
    so uproot.open("osdf://...") fails: read over HTTPS then, and say how to fix it."""
    from importlib.metadata import version
    try:
        pel = tuple(int(x) for x in version("pelicanfs").split(".")[:3])
        dav = int(version("aiowebdav2").split(".")[0])
    except Exception:                                # noqa: BLE001
        return False
    if pel <= (1, 5, 0) and dav >= 1:
        print(f"note: pelicanfs {version('pelicanfs')} does not work with aiowebdav2 "
              f"{version('aiowebdav2')}; reading over HTTPS instead.  To fix: "
              f"pip install 'aiowebdav2<1'", file=sys.stderr)
        return True
    return False


def backend() -> str:
    """How the files are read: "pelicanfs <version>" or "HTTPS (standard library)"."""
    if osdf_fs() is None:
        return "HTTPS (standard library)"
    from importlib.metadata import version
    return f"pelicanfs {version('pelicanfs')}"


def _pelican(call, path: str):
    """``call(fs)`` with pelicanfs's file system, again (at the next cache) when it fails."""
    for attempt in range(RETRIES):
        try:
            return call(osdf_fs())
        except (FileNotFoundError, PermissionError):
            raise
        except Exception as e:                       # noqa: BLE001
            if attempt == RETRIES - 1:
                raise
            if type(e).__name__ == "NoAvailableSource":   # every cache failed
                wait = RETRY_WAIT * 2 ** min(attempt, 3)
                print(f"  {_name(path)}: no cache left; asking the director again in {wait} s",
                      file=sys.stderr)
                time.sleep(wait)
                osdf_fs(fresh=True)
            else:
                print(f"  {_name(path)}: {type(e).__name__}: {e}; trying the next cache",
                      file=sys.stderr)


def read_bytes(cfg: dict, set_name: str, filename: str) -> bytes:
    """A whole (small) file of a set: with pelicanfs, else over HTTPS."""
    if osdf_fs() is not None:
        path = osdf_path(cfg, set_name, filename)
        return _pelican(lambda fs: fs.cat_file(path), path)
    with _open(url_of(cfg, set_name, filename)) as r:
        return r.read()


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
    files = json.loads(read_bytes(cfg, set_name, MANIFEST))["files"]
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
    """``path`` relative to the current directory when it is below it, else with the home
    directory written as ~ (output gets shared in notebooks)."""
    try:
        return str(Path(path).absolute().relative_to(Path.cwd()))
    except ValueError:
        pass
    s, home = str(path), str(Path.home())
    return "~" + s[len(home):] if home not in ("", "/") and s.startswith(home) else s


def _fmt(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1000
    return ""


def _progress(dest: Path, done: int, size: int, quiet: bool) -> None:
    if not quiet and sys.stdout.isatty():
        print(f"\r  ...     {dest.name}  {_fmt(done)} / {_fmt(size)}", end="")


def _fetch_pelican(path: str, part: Path, size: int, quiet: bool) -> None:
    """Read ``path`` into ``part`` with pelicanfs, BLOCK bytes at a time, from where the
    .part ends (a failed read goes to the next cache)."""
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    with open(part, "ab" if have else "wb") as out:
        pos = have
        while pos < size:
            end = min(pos + BLOCK, size)
            data = _pelican(lambda fs: fs.cat_file(path, start=pos, end=end), path)
            if not data:
                raise IOError(f"{_name(path)}: no data at byte {pos} of {size}")
            out.write(data)
            pos += len(data)
            _progress(part.with_suffix(""), pos, size, quiet)


def fetch_file(url: str, dest: Path, size: int, crc: str | None = None,
               quiet: bool = False, osdf: str | None = None) -> Path:
    """Download ``url`` to ``dest`` (resuming a .part), check the size and, if possible, the
    CRC32C.  A complete file already there is kept.  ``osdf``: the file's OSDF name, read
    with pelicanfs when it is installed (else ``url`` over HTTPS)."""
    if dest.exists() and dest.stat().st_size == size:
        if not quiet:
            print(f"  ok      {dest.name}  ({_fmt(size)}, already there)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    t0, start = time.time(), (part.stat().st_size if part.exists() else 0)
    if osdf and osdf_fs() is not None:
        try:
            _fetch_pelican(osdf, part, size, quiet)
        except (FileNotFoundError, PermissionError):
            raise
        except Exception as e:                       # noqa: BLE001
            print(f"  {dest.name}: pelicanfs failed ({type(e).__name__}: {e}); "
                  f"going on over HTTPS", file=sys.stderr)
    done_already = part.exists() and part.stat().st_size == size
    for attempt in range(0 if done_already else RETRIES):   # over HTTPS: a connection
        have = part.stat().st_size if part.exists() else 0    # dropped mid-file resumes
        # from .part, at the next cache
        if have > size:
            part.unlink()
            have = 0
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
                        _progress(dest, done, size, quiet)
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
        print(f"{set_name}: {len(files)} file(s), {_fmt(total)} -> {_short(dest)}"
              f"  [{backend()}]")
    for name, meta in files.items():
        if dry_run:
            print(f"  would fetch {name}  ({_fmt(meta['size'])})")
            continue
        fetch_file(url_of(cfg, set_name, name), dest / name, meta["size"],
                   meta.get("crc32c"), quiet=quiet, osdf=osdf_path(cfg, set_name, name))
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
