"""
utils/remote_transfer/transfer_core.py

What js_gcs.py (Google Cloud Storage) and js_osdf.py (Pelican/OSDF) share: the kinds of
production files (--what), the upload / download / ls commands with their skip and check
rules, the parallel transfers, and the environment each script makes for itself.  Only the
standard library at import: the scripts import this module before they are in their
environment.

A store (GcsStore, OsdfStore) gives the commands:

    label                  'gs://test_fno', 'osdf:///fno4hic': the root, for messages
    relative(text)         'gs://test_fno/x/y' or 'x/y' -> 'x/y'
    list(prefix)           {name: (size, crc32c or None)} of every file below a directory
                           ('': all)
    stat(name)             (size, crc32c or None), or None if there is no such file
    top_level(prefix)      (directories, {name: (size, crc32c)}) directly below prefix
    upload(path, name)     local file -> name
    download(name, path)   name -> local file path
    read_bytes(name)       the content of a small file, None if there is none
    native_checksums       True: the store keeps a CRC32C of every object and checks
                           transfers against it (GCS)
    manifest               True: no CRC32C from the store, so one is kept in a manifest
                           file per directory (MANIFEST), written after each upload

With a manifest, uploads compute the file's CRC32C, check the size the store reports
afterwards, and record both; downloads and the skip check use the recorded CRC32C.  Files
the manifest doesn't know (put there by other tools) are compared by size.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROG = "js_transfer"                      # set by main(): the script's name, for messages
KINDS = ("pair", "h5", "root", "other")
HADRON_TAGS = ("bulk_jet", "bulk_bg", "jet_frag")
WILDCARDS = "*?["
MANIFEST = ".js_transfer_crc32c.json"


def die(msg):
    sys.exit(f"{PROG}: {msg}")


def say(msg):
    print(f"{PROG}: {msg}", flush=True)


# ── its own environment ────────────────────────────────────────────────────────────
class Env:
    """The virtual environment a script makes for itself (~/.cache/<name>/venv)."""

    def __init__(self, name, requirements, env_var):
        self.name, self.requirements = name, tuple(requirements)
        self.dir = Path(os.environ.get(env_var, f"~/.cache/{name}/venv")).expanduser()
        self.stamp = self.dir / f".{name}_requirements"

    @property
    def python(self):
        return self.dir / "bin" / "python"

    def active(self):
        try:
            return Path(sys.prefix).resolve() == self.dir.resolve()
        except OSError:
            return False

    def ok(self):
        # resolve(): the venv's python links to the one it was made from, which may be gone
        py = self.python
        return (py.exists() and py.resolve().exists() and self.stamp.exists()
                and self.stamp.read_text() == "\n".join(self.requirements))

    def ensure(self, reinstall=False):
        """Make (or remake) the environment; returns its python."""
        if self.ok() and not reinstall:
            return self.python
        if self.dir.exists():
            shutil.rmtree(self.dir)
        print(f"{self.name}: making its environment in {self.dir} (once) ...", flush=True)
        self.dir.parent.mkdir(parents=True, exist_ok=True)
        if subprocess.run([sys.executable, "-m", "venv", str(self.dir)],
                          check=False).returncode != 0:
            sys.exit(f"{self.name}: python3 -m venv failed (on Debian/Ubuntu: apt install "
                     "python3-venv)")
        py = str(self.python)
        for cmd in ([py, "-m", "pip", "install", "-q", "--upgrade", "pip"],
                    [py, "-m", "pip", "install", "-q", *self.requirements]):
            if subprocess.run(cmd, check=False).returncode != 0:
                shutil.rmtree(self.dir, ignore_errors=True)
                sys.exit(f"{self.name}: {' '.join(cmd[3:])} failed")
        self.stamp.write_text("\n".join(self.requirements))
        print(f"{self.name}: environment ready", flush=True)
        return self.python

    def remove(self):
        """Delete the environment (only a venv this script made: pyvenv.cfg + its stamp)."""
        if not self.dir.exists():
            print(f"{self.name}: no environment at {self.dir}")
            return 0
        if not ((self.dir / "pyvenv.cfg").is_file() and self.stamp.is_file()):
            print(f"{self.name}: {self.dir} is not an environment made by {self.name}.py: "
                  "left alone")
            return 1
        shutil.rmtree(self.dir)
        try:
            self.dir.parent.rmdir()                    # ~/.cache/<name>, if now empty
        except OSError:
            pass
        print(f"{self.name}: removed {self.dir} (the next run makes it again)")
        return 0


def enter_env(env, script, argv, no_env_var):
    """At the start of a script: setup --remove, or run the script again in its environment
    (made first if needed), unless already there or $<no_env_var> = 1."""
    if argv[:1] == ["setup"] and "--remove" in argv[1:]:
        sys.exit(env.remove())                         # before (not inside) the environment
    if env.active() or os.environ.get(no_env_var) == "1":
        return
    py = env.ensure(reinstall=argv[:1] == ["setup"] and "--reinstall" in argv[1:])
    os.execv(str(py), [str(py), os.path.abspath(script), *argv])


# ── what to transfer ──────────────────────────────────────────────────────────────
def parse_what(text):
    """'pair,root' -> {'pair', 'root'}; 'all' -> every kind."""
    out = set()
    for w in (x.strip() for x in text.split(",") if x.strip()):
        if w == "all":
            out |= set(KINDS)
        elif w in KINDS[:3]:
            out.add(w)
        else:
            raise argparse.ArgumentTypeError(f"{w!r} is not pair, h5, root or all")
    if not out:
        raise argparse.ArgumentTypeError("empty")
    return out


def kind_of(name, names):
    """The kind of file ``name`` (a base name), given every base name in its directory
    ``names``; None for files never transferred (*.part, hidden, the manifest)."""
    if name.startswith(".") or name.endswith((".part", ".tmp")):
        return None
    if any(name.endswith(f"_hadrons_{t}.h5") for t in HADRON_TAGS) or name.endswith(
            ("_particlize.h5", "_hadronize.log")):
        return "h5"
    if name.endswith(("_hadrons.root", "_campaign.root")):
        return "root"
    stem, ext = os.path.splitext(name)
    # a pair file has the run's .json / .xml next to it (or was particlized); that tells
    # it from other .h5 in a production directory (wake_observables.h5, ...)
    if ext in (".h5", ".json", ".xml", ".log") and f"{stem}.h5" in names and any(
            f"{stem}{s}" in names for s in (".json", ".xml", "_particlize.h5")):
        return "pair"
    return "other"


def classify(relpaths):
    """{relative path: kind} for the paths of one listing (kinds per directory)."""
    by_dir = {}
    for p in relpaths:
        by_dir.setdefault(os.path.dirname(p), set()).add(os.path.basename(p))
    out = {}
    for p in relpaths:
        k = kind_of(os.path.basename(p), by_dir[os.path.dirname(p)])
        if k is not None:
            out[p] = k
    return out


def h5_incomplete(path):
    """True if an HDF5 file says it is not complete (attribute ``complete`` False), or
    cannot be opened (still being written); files without the attribute count as complete."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            c = f.attrs.get("complete")
    except OSError:
        return True
    return c is not None and not bool(c)


def crc32c_b64(path, block=16 * 1024 * 1024):
    """A file's CRC32C as GCS writes it (base64 of the big-endian 4 bytes)."""
    import google_crc32c
    c = google_crc32c.Checksum()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            c.update(chunk)
    return base64.b64encode(c.digest()).decode()


def same(path, remote, size_only):
    """Local file ``path`` and remote (size, crc32c or None) hold the same bytes."""
    if remote is None or not path.exists() or path.stat().st_size != remote[0]:
        return False
    return size_only or remote[1] is None or crc32c_b64(path) == remote[1]


def fmt_size(n):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return ""


# ── manifests (stores without checksums) ──────────────────────────────────────────
def _manifest_name(directory):
    return f"{directory}/{MANIFEST}" if directory else MANIFEST


def read_manifest(store, directory):
    """{base name: {"size", "crc32c", ...}} of a remote directory ({} if none)."""
    data = store.read_bytes(_manifest_name(directory))
    if not data:
        return {}
    try:
        return dict(json.loads(data).get("files", {}))
    except (ValueError, AttributeError):
        say(f"warning: {store.label}/{_manifest_name(directory)} is not readable: ignored")
        return {}


def with_manifest(store, listing, cache):
    """Fill in the CRC32C of listed files from their directories' manifests (where the
    manifest's size is the listed one).  cache: {directory: manifest}, filled as needed."""
    if not store.manifest:
        return listing
    out = {}
    for name, (size, crc) in listing.items():
        d, base = os.path.dirname(name), os.path.basename(name)
        if d not in cache:
            cache[d] = read_manifest(store, d)
        m = cache[d].get(base)
        out[name] = (size, m["crc32c"] if m and m.get("size") == size else crc)
    return out


def write_manifests(store, entries):
    """Add {name: {"size", "crc32c"}} to the manifests of their directories (read again
    first, so the entries of other uploads there are kept)."""
    by_dir = {}
    for name, e in entries.items():
        by_dir.setdefault(os.path.dirname(name), {})[os.path.basename(name)] = e
    for d, files in by_dir.items():
        m = read_manifest(store, d)
        m.update(files)
        doc = {"format": "js_transfer crc32c manifest", "crc32c": "base64, as GCS",
               "files": dict(sorted(m.items()))}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(doc, f, indent=1)
        try:
            store.upload(Path(f.name), _manifest_name(d))
        finally:
            os.unlink(f.name)


# ── transfers ─────────────────────────────────────────────────────────────────────
def _run(jobs, n_workers, label):
    """jobs: {name: (size, callable)}.  Runs them, a line per file; returns (failed names,
    {name: return value}, interrupted)."""
    t0, done, failed, results = time.time(), 0, [], {}
    total = sum(s for s, _ in jobs.values())

    def timed(fn):
        t = time.time()
        return fn(), time.time() - t

    interrupted = False
    ex = ThreadPoolExecutor(max_workers=n_workers)
    futs = {ex.submit(timed, fn): (name, size) for name, (size, fn) in jobs.items()}
    try:
        for fut in as_completed(futs):
            name, size = futs[fut]
            try:
                results[name], dt = fut.result()
            except Exception as e:                     # noqa: BLE001
                failed.append(name)
                print(f"  FAILED  {name}: {type(e).__name__}: {e}", flush=True)
                continue
            done += size
            print(f"  {label}  {fmt_size(size):>9}  {name}  ({fmt_size(size / max(dt, 1e-3))}"
                  f"/s; {fmt_size(done)} of {fmt_size(total)})", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        ex.shutdown(wait=False, cancel_futures=True)
    else:
        ex.shutdown()
    dt = time.time() - t0
    say(f"{len(results)} file(s), {fmt_size(done)} in {dt:.0f} s "
        f"({fmt_size(done / max(dt, 1e-3))}/s)" + (f", {len(failed)} FAILED" if failed else "")
        + (", INTERRUPTED: re-run the same command to finish" if interrupted else ""))
    return failed, results, interrupted


def _summary(kinds_of, sizes, what):
    by = {}
    for p, k in kinds_of.items():
        if k in what:
            n, s = by.get(k, (0, 0))
            by[k] = (n + 1, s + sizes[p])
    return ", ".join(f"{k} {n} file(s) {fmt_size(s)}" for k, (n, s) in sorted(by.items()))


def _plan(jobs, skipped, a, verb, where):
    """Print the plan; True if there is something to transfer (and no --dry-run)."""
    say(f"{len(jobs)} to {verb} ({fmt_size(sum(s for s, _ in jobs.values()))}), "
        f"{skipped} already {where}")
    if a.dry_run:
        for name, (size, _) in jobs.items():
            print(f"  would {verb}  {fmt_size(size):>9}  {name}")
        return False
    return bool(jobs)


def match(pattern, store):
    """({name: (size, crc32c)} matching a pattern (* ? [..]; * also matches /), and their
    kinds, judged among all files listed with them: a pair file's .json is 'pair' even when
    the pattern matches it alone)."""
    import fnmatch
    base = pattern[:min(pattern.find(c) for c in WILDCARDS if c in pattern)]
    listing = store.list(base.rsplit("/", 1)[0] if "/" in base else "")
    kinds = classify(list(listing))
    found = {n: v for n, v in listing.items() if fnmatch.fnmatchcase(n, pattern)}
    return found, {n: kinds[n] for n in found if n in kinds}


def send(store, path, name):
    """Upload one file; with a manifest, its CRC32C first and the stored size after."""
    size = path.stat().st_size
    crc = crc32c_b64(path) if store.manifest else None
    store.upload(path, name)
    if store.manifest:
        st = store.stat(name)
        if st is None or st[0] != size:
            raise OSError(f"stored size {st and st[0]} is not the file's {size}")
        return {"size": size, "crc32c": crc,
                "uploaded": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    return None


def fetch(store, name, path, remote):
    """Download one file via <path>.part: its size, and its CRC32C where the store doesn't
    check it itself, must be remote's."""
    tmp = path.with_name(path.name + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        store.download(name, tmp)
        size = tmp.stat().st_size
        if size != remote[0]:
            raise OSError(f"got {size} bytes, not {remote[0]}")
        if remote[1] and not store.native_checksums and crc32c_b64(tmp) != remote[1]:
            raise OSError("CRC32C differs from the manifest's")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


# ── commands ──────────────────────────────────────────────────────────────────────
def cmd_upload(a, store):
    jobs, skipped, incomplete, cache = {}, 0, [], {}
    prefix = a.prefix.strip("/")

    def consider(path, name, remote):
        nonlocal skipped
        if path.name.endswith(".h5") and not a.include_incomplete and h5_incomplete(path):
            incomplete.append(str(path))
        elif not a.force and same(path, remote, a.size_only):
            skipped += 1
        else:
            jobs[name] = (path.stat().st_size, lambda p=path, n=name: send(store, p, n))

    dirs, files = [], {}
    for p in a.paths:
        p = Path(p).expanduser().resolve()
        if p.is_dir():
            dirs.append(p)
        elif p.is_file():
            files.setdefault(p.parent, []).append(p.name)
        else:
            die(f"{p} not found")
    rename = a.rename.strip("/") if a.rename else None
    if rename is not None:                             # --as: one local directory only
        if not rename:
            die("--as needs a name")
        if a.flat:
            die("--as and --flat don't go together (--as names the folder, --flat drops it)")
        if len(set(dirs) | set(files)) > 1:
            die("--as names one folder: give it the files of one directory only")

    def folder(d):
        return "/".join(x for x in (prefix, rename or d.name) if x)

    # directories: their files of the selected kinds
    for d in dirs:
        kinds = classify(sorted(p.name for p in d.iterdir() if p.is_file()))
        dest = folder(d)
        remote = with_manifest(store, store.list(dest), cache)
        sizes = {f: (d / f).stat().st_size for f in kinds}
        say(f"{d} -> {store.label}/{dest}/: "
            f"{_summary(kinds, sizes, a.what) or 'nothing selected'}")
        for f, k in kinds.items():
            if k in a.what:
                consider(d / f, f"{dest}/{f}", remote.get(f"{dest}/{f}"))
    # files: each one named, whatever its kind; where its directory's upload puts it
    for parent, names in files.items():
        kinds = classify(sorted(p.name for p in parent.iterdir() if p.is_file()))
        dest = prefix if a.flat else folder(parent)
        for f in names:
            if kinds.get(f) is None:
                say(f"{parent / f} left out (temporary or hidden file)")
                continue
            name = f"{dest}/{f}" if dest else f
            say(f"{parent / f} ({kinds[f]}) -> {store.label}/{name}")
            st = store.stat(name)
            consider(parent / f, name,
                     with_manifest(store, {name: st}, cache)[name] if st else None)
    if incomplete:
        say(f"{len(incomplete)} incomplete HDF5 file(s) left out (--include-incomplete to "
            "upload them): " + ", ".join(os.path.basename(p) for p in incomplete[:5])
            + (" ..." if len(incomplete) > 5 else ""))
    if not _plan({f"{store.label}/{n}": v for n, v in jobs.items()}, skipped, a, "upload",
                 "there"):
        return 0
    failed, results, interrupted = _run(jobs, a.jobs, "up  ")
    if store.manifest and results:
        write_manifests(store, results)
    if interrupted:
        sys.exit(130)
    return 1 if failed else 0


def cmd_download(a, store):
    to = Path(a.to).expanduser().resolve()
    want, cache = {}, {}                               # local path -> (name, (size, crc))

    def add(path, name, v):
        if want.setdefault(path, (name, v))[0] != name:
            die(f"{want[path][0]} and {name} would both be {path}: download them separately "
                "(--to)")

    for target in a.targets:
        t = store.relative(target)
        if not t:
            die("download needs a directory, file or pattern, not the whole "
                f"{store.label}")
        if any(c in t for c in WILDCARDS):             # a pattern: files, to TO/<file>
            found, kinds = match(t, store)
            sel = {n: v for n, v in found.items() if kinds.get(n) in a.what}
            say(f"{store.label}/{t} -> {to}/: {len(sel)} of {len(found)} matching file(s) "
                "selected")
            for n, v in sel.items():
                add(to / os.path.basename(n), n, v)
        elif (v := store.stat(t)) is not None:         # one file, to TO/<file>
            say(f"{store.label}/{t} -> {to}/")
            add(to / os.path.basename(t), t, v)
        else:                                          # a directory, to TO/<last part>/
            remote = store.list(t)
            if not remote:
                die(f"nothing at {store.label}/{t}")
            base = os.path.dirname(t)                  # .../NAME/x -> TO/NAME/x
            rel = {n: os.path.relpath(n, base) if base else n for n in remote}
            kinds = classify(list(rel.values()))
            sizes = {rel[n]: remote[n][0] for n in remote}
            say(f"{store.label}/{t}/ -> {to / os.path.basename(t)}/: "
                f"{_summary(kinds, sizes, a.what) or 'nothing selected'}")
            for n in sorted(remote):
                if kinds.get(rel[n]) in a.what:
                    add(to / rel[n], n, remote[n])
    checks = with_manifest(store, {n: v for n, v in want.values()}, cache)
    names = [n for n, _ in want.values()]
    jobs, skipped = {}, 0
    for path, (n, _) in sorted(want.items(), key=lambda x: x[1][0]):
        v = checks[n]
        if not a.force and same(path, v, a.size_only):
            skipped += 1
        else:                                          # one object to two places: say where
            key = n if names.count(n) == 1 else f"{n} -> {path}"
            jobs[key] = (v[0], lambda n=n, p=path, v=v: fetch(store, n, p, v))
    if not _plan(jobs, skipped, a, "download", "here"):
        return 0
    failed, _, interrupted = _run(jobs, a.jobs, "down")
    if interrupted:
        sys.exit(130)
    return 1 if failed else 0


def cmd_ls(a, store):
    prefix = store.relative(a.prefix) if a.prefix else ""
    if any(c in prefix for c in WILDCARDS):
        remote, kinds = match(prefix, store)
    elif not a.recursive and a.what == set(KINDS):
        dirs, files = store.top_level(prefix)
        for d in dirs:
            print(f"  {'':>9}  {store.label}/{d}")
        for n, (s, _) in sorted(files.items()):
            if os.path.basename(n) != MANIFEST:
                print(f"  {fmt_size(s):>9}  {store.label}/{n}")
        if not dirs and not files:
            say(f"nothing under {store.label}/{prefix}")
        return 0
    else:
        remote = store.list(prefix)
        kinds = classify(list(remote))
    sizes = {n: s for n, (s, _) in remote.items()}
    for n in sorted(remote):
        if kinds.get(n) in a.what:
            print(f"  {fmt_size(sizes[n]):>9}  {kinds[n]:<5}  {store.label}/{n}")
    say(_summary(kinds, sizes, a.what) or "nothing")
    return 0


# ── command line ──────────────────────────────────────────────────────────────────
def build_parser(doc, add_store_args, root_help):
    """The common command line; add_store_args(parser) adds the store's own options."""
    p = argparse.ArgumentParser(description=doc,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_store_args(p)
    sub = p.add_subparsers(dest="cmd", required=True)

    def transfer_opts(q):
        q.add_argument("--what", type=parse_what, default=set(KINDS),
                       help="pair, h5, root or all, or a comma list (default all)")
        q.add_argument("-j", "--jobs", type=int, default=4, help="files at once (default 4)")
        q.add_argument("--dry-run", action="store_true", dest="dry_run",
                       help="show what would be transferred")
        q.add_argument("--force", action="store_true", help="transfer files already there")
        q.add_argument("--size-only", action="store_true", dest="size_only",
                       help="files of the same size count as the same (no CRC32C of the "
                            "local file)")

    up = sub.add_parser("upload", help=f"directories or files -> {root_help}/[PREFIX/]"
                                       "<dir name>/<file>")
    up.add_argument("paths", nargs="+", metavar="PATH",
                    help="production directories (their files of the --what kinds) and "
                         "files (each one, whatever its kind)")
    up.add_argument("--prefix", default="", help="put everything below this prefix")
    up.add_argument("--as", dest="rename", metavar="NAME", default=None,
                    help=f"the remote folder's name instead of the local directory's: "
                         f"{root_help}/[PREFIX/]NAME/<file> (sources from one directory)")
    up.add_argument("--flat", action="store_true",
                    help=f"files go to {root_help}/[PREFIX/]<file>, without their "
                         "directory's name (directories are not affected)")
    up.add_argument("--include-incomplete", action="store_true", dest="include_incomplete",
                    help="also HDF5 files whose 'complete' attribute is False")
    transfer_opts(up)
    down = sub.add_parser("download", help="directories, files or patterns -> TO/")
    down.add_argument("targets", nargs="+", metavar="TARGET",
                      help="a directory (-> TO/<its name>/), a file or a pattern with * ? "
                           f"[..] (-> TO/<file>); NAME or {root_help}/NAME; quote patterns")
    down.add_argument("--to", default=".", help="local directory (default .)")
    transfer_opts(down)
    ls = sub.add_parser("ls", help="list, at the top or in a directory")
    ls.add_argument("prefix", nargs="?", default="", help="a directory, or a pattern")
    ls.add_argument("--what", type=parse_what, default=set(KINDS),
                    help="pair, h5, root or all (lists recursively, with the kinds)")
    ls.add_argument("-r", "--recursive", action="store_true", help="every file below")
    st = sub.add_parser("setup", help="make the environment (--reinstall: again; --remove: "
                        "delete it)")
    g = st.add_mutually_exclusive_group()
    g.add_argument("--reinstall", action="store_true", help="make it again")
    g.add_argument("--remove", action="store_true", help="delete it")
    return p


def main(prog, parser, env, make_store, required_modules, argv=None):
    """Run a command: parse, check the environment, make the store, dispatch."""
    global PROG
    PROG = prog
    a = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if getattr(a, "jobs", 1) < 1:
        parser.error("-j must be >= 1")
    if a.cmd == "setup":
        print(f"{prog}: environment {env.dir} ({sys.executable}): "
              + ", ".join(env.requirements))
        return 0
    try:
        for m in required_modules:
            __import__(m)
    except ImportError as e:
        die(f"{e}: the environment {env.dir} is broken; run ./{prog}.py setup --reinstall")
    store = make_store(a)
    try:
        return {"upload": cmd_upload, "download": cmd_download, "ls": cmd_ls}[a.cmd](a, store)
    except (OSError, ConnectionError, TimeoutError, RuntimeError) as e:
        die(f"{store.label}: {describe(e)}")
    except Exception as e:                             # noqa: BLE001  the client's own errors
        if type(e).__module__.split(".")[0] in ("builtins", "transfer_core"):
            raise
        die(f"{store.label}: {describe(e)}")


def describe(e):
    """An exception and, if it has one, its root cause (an expired certificate, ...)."""
    root = e
    while root.__cause__ is not None or root.__context__ is not None:
        root = root.__cause__ or root.__context__
    text = f"{type(e).__name__}: {e}"
    return text if root is e else f"{text} ({type(root).__name__}: {root})"
