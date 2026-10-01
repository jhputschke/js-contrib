#!/usr/bin/env python3
"""
utils/gcs_transfer/js_gcs.py

Upload and download production directories to / from Google Cloud Storage, with a
service-account key.  Standalone: on its first run the script makes its own virtual
environment (``~/.cache/js_gcs/venv``, plain ``python3 -m venv`` + pip) and from then on
runs itself in it, whatever environment it is started from.

    ./js_gcs.py upload /data/AuAu_0_10_pth10-40 --what root   # -> gs://test_fno/AuAu_0_10_pth10-40/
    ./js_gcs.py upload /data/AuAu_a /data/AuAu_b --what pair,h5 --prefix productions -j 8
    ./js_gcs.py download AuAu_0_10_pth10-40 --what root --to /scratch
                                                       # -> /scratch/AuAu_0_10_pth10-40/
    ./js_gcs.py upload /data/AuAu_a/AuAu_a_0003_hadrons.root /data/AuAu_a/*_0004*.h5
                                                       # files -> gs://test_fno/AuAu_a/<file>
    ./js_gcs.py download AuAu_a/AuAu_a_0003_hadrons.root 'AuAu_a/*_0004_*' --to here
                                                       # files -> here/<file>
    ./js_gcs.py ls                                                  # the bucket's top level
    ./js_gcs.py ls AuAu_0_10_pth10-40 --what h5                     # files, sizes, kinds
    ./js_gcs.py setup --reinstall                                   # (re)make the environment
    ./js_gcs.py setup --remove                                      # delete the environment

What (``--what``, a comma list; default all) -- by file name, as run_prod_jet.py,
hadronize.py and run_h5toROOT.py name them:

    pair    <stem>.h5, the hydro pair file, with its <stem>.json, .xml, .log
    h5      <stem>_particlize.h5, <stem>_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5,
            <stem>_hadronize.log
    root    <stem>_hadrons.root, <campaign>_campaign.root
    all     the three, plus every other file of the directory (run_jobs.campaign,
            wake_observables.h5, ...)

Uploads go to gs://BUCKET/[PREFIX/]<directory name>/<file>: for a directory, its files of
the --what kinds (not its subdirectories); for a file named on the command line, that file
whatever its kind, to where its directory's upload puts it (--flat: [PREFIX/]<file>).
Downloads of a directory go to TO/<its name>/..., of a file or of a pattern (* ? [..],
quoted; * also matches /) to TO/<file>.  A file already there with the same size and CRC32C
is skipped (--size-only: the size alone; --force: transfer anyway), so re-running a command
finishes an interrupted transfer.  HDF5 files whose ``complete`` attribute is False (a job
still writing) are not uploaded, unless --include-incomplete; ``*.part`` files never.
Every transfer is checked against the CRC32C; downloads go to ``<file>.part`` first and are
renamed when complete.

The key: --key-file, else $JS_GCS_KEY_FILE, else $GOOGLE_APPLICATION_CREDENTIALS, else
fnotest-wayne-gcs.json in the current directory, next to this script, or in
~/.config/js_gcs/.  The bucket: --bucket, else $JS_GCS_BUCKET, else gs://test_fno.
"""

from __future__ import annotations

import argparse
import base64
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── its own environment ────────────────────────────────────────────────────────────
REQUIREMENTS = ("google-cloud-storage>=2.14", "google-crc32c>=1.5", "h5py>=3.0")
ENV_DIR = Path(os.environ.get("JS_GCS_ENV", "~/.cache/js_gcs/venv")).expanduser()
STAMP = ".js_gcs_requirements"


def _env_python():
    return ENV_DIR / "bin" / "python"


def _in_env():
    try:
        return Path(sys.prefix).resolve() == ENV_DIR.resolve()
    except OSError:
        return False


def _env_ok():
    py = _env_python()
    stamp = ENV_DIR / STAMP
    # resolve(): the venv's python is a link to the one it was made from, which may be gone
    return (py.exists() and py.resolve().exists() and stamp.exists()
            and stamp.read_text() == "\n".join(REQUIREMENTS))


def ensure_env(reinstall=False):
    """Make (or remake) the environment; returns its python."""
    if _env_ok() and not reinstall:
        return _env_python()
    if ENV_DIR.exists():
        shutil.rmtree(ENV_DIR)
    print(f"js_gcs: making its environment in {ENV_DIR} (once) ...", flush=True)
    ENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-m", "venv", str(ENV_DIR)], check=False)
    if r.returncode != 0:
        sys.exit("js_gcs: python3 -m venv failed (on Debian/Ubuntu: apt install python3-venv)")
    py = str(_env_python())
    for cmd in ([py, "-m", "pip", "install", "-q", "--upgrade", "pip"],
                [py, "-m", "pip", "install", "-q", *REQUIREMENTS]):
        if subprocess.run(cmd, check=False).returncode != 0:
            shutil.rmtree(ENV_DIR, ignore_errors=True)
            sys.exit(f"js_gcs: {' '.join(cmd[3:])} failed")
    (ENV_DIR / STAMP).write_text("\n".join(REQUIREMENTS))
    print("js_gcs: environment ready", flush=True)
    return _env_python()


def remove_env():
    """Delete the environment (only a venv this script made: pyvenv.cfg + its stamp)."""
    if not ENV_DIR.exists():
        print(f"js_gcs: no environment at {ENV_DIR}")
        return 0
    if not ((ENV_DIR / "pyvenv.cfg").is_file() and (ENV_DIR / STAMP).is_file()):
        print(f"js_gcs: {ENV_DIR} is not an environment made by js_gcs.py: left alone")
        return 1
    shutil.rmtree(ENV_DIR)
    try:
        ENV_DIR.parent.rmdir()                         # ~/.cache/js_gcs, if now empty
    except OSError:
        pass
    print(f"js_gcs: removed {ENV_DIR} (the next run makes it again)")
    return 0


def reexec_in_env(argv):
    """Run this script again in its environment (making it first if needed)."""
    py = ensure_env(reinstall="--reinstall" in argv and argv[:1] == ["setup"])
    os.execv(str(py), [str(py), os.path.abspath(__file__), *argv])


# ── what to transfer ──────────────────────────────────────────────────────────────
KINDS = ("pair", "h5", "root", "other")
HADRON_TAGS = ("bulk_jet", "bulk_bg", "jet_frag")
DEFAULT_BUCKET = "test_fno"
KEY_NAME = "fnotest-wayne-gcs.json"


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
    ``names``; None for files never transferred (*.part, hidden)."""
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


# ── GCS ───────────────────────────────────────────────────────────────────────────
def find_key(arg):
    """The service-account key file, or exit with where it was looked for."""
    here = Path(__file__).resolve().parent
    cands = [arg, os.environ.get("JS_GCS_KEY_FILE"),
             os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
             Path.cwd() / KEY_NAME, here / KEY_NAME,
             Path("~/.config/js_gcs").expanduser() / KEY_NAME]
    for c in cands[:3]:
        if c:                                       # given explicitly: must exist
            if not Path(c).expanduser().is_file():
                sys.exit(f"js_gcs: key file {c} not found")
            return Path(c).expanduser()
    for c in cands[3:]:
        if c.is_file():
            return c
    where = ", ".join(dict.fromkeys(str(c.parent) for c in cands[3:]))
    sys.exit(f"js_gcs: no key file: pass --key-file, set JS_GCS_KEY_FILE, or put {KEY_NAME} "
             f"in one of {where}")


def bucket_name(text):
    return text.strip().removeprefix("gs://").strip("/")


class Gcs:
    """One bucket, a client per thread (google-cloud-storage clients aren't shared)."""

    def __init__(self, bucket, key):
        self.bucket_name, self.key = bucket, str(key)
        self._local = threading.local()

    def bucket(self):
        if not hasattr(self._local, "bucket"):
            from google.cloud import storage
            client = storage.Client.from_service_account_json(self.key)
            self._local.bucket = client.bucket(self.bucket_name)
        return self._local.bucket

    def list(self, prefix, directory=True):
        """{object name: (size, crc32c)} below the directory ``prefix`` (directory=False:
        every name that starts with ``prefix``)."""
        p = prefix.strip("/") if directory else prefix
        if directory and p:
            p += "/"
        it = self.bucket().client.list_blobs(self.bucket_name, prefix=p or None)
        return {b.name: (int(b.size), b.crc32c) for b in it if not b.name.endswith("/")}

    def stat(self, name):
        """(size, crc32c) of object ``name``, None if there is none."""
        b = self.bucket().get_blob(name)
        return None if b is None else (int(b.size), b.crc32c)

    def top_level(self, prefix):
        """The 'directories' and files directly below ``prefix``."""
        p = prefix.strip("/")
        it = self.bucket().client.list_blobs(self.bucket_name, prefix=f"{p}/" if p else None,
                                             delimiter="/")
        files = {b.name: (int(b.size), b.crc32c) for b in it}
        return sorted(it.prefixes), files

    def upload(self, path, name):
        from google.cloud.storage.retry import DEFAULT_RETRY
        blob = self.bucket().blob(name, chunk_size=64 * 1024 * 1024)
        # retried although not idempotent in general: we write the same content again
        blob.upload_from_filename(str(path), checksum="crc32c", timeout=600,
                                  retry=DEFAULT_RETRY)

    def download(self, name, path):
        blob = self.bucket().blob(name)
        tmp = Path(f"{path}.part")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(tmp), checksum="crc32c", timeout=600)
        os.replace(tmp, path)


def crc32c_b64(path, block=16 * 1024 * 1024):
    """A file's CRC32C as GCS stores it (base64 of the big-endian 4 bytes)."""
    import google_crc32c
    c = google_crc32c.Checksum()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            c.update(chunk)
    return base64.b64encode(c.digest()).decode()


def same(path, remote, size_only):
    """Local file ``path`` and remote (size, crc32c) hold the same bytes."""
    if remote is None or not path.exists() or path.stat().st_size != remote[0]:
        return False
    return size_only or remote[1] is None or crc32c_b64(path) == remote[1]


def fmt_size(n):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


# ── commands ──────────────────────────────────────────────────────────────────────
def _run(jobs, n_workers, label):
    """jobs: [(size, name, callable)]; runs them, prints a line per file, returns failures."""
    t0, done, failed = time.time(), 0, []
    total = sum(s for s, _, _ in jobs)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(fn): (size, name, time.time()) for size, name, fn in jobs}
        try:
            for fut in as_completed(futs):
                size, name, t = futs[fut]
                try:
                    fut.result()
                except Exception as e:                       # noqa: BLE001
                    failed.append(name)
                    print(f"  FAILED  {name}: {type(e).__name__}: {e}", flush=True)
                    continue
                done += size
                dt = max(time.time() - t, 1e-3)
                print(f"  {label}  {fmt_size(size):>9}  {name}  ({fmt_size(size / dt)}/s; "
                      f"{fmt_size(done)} of {fmt_size(total)})", flush=True)
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            sys.exit("\njs_gcs: interrupted; re-run the same command to finish")
    dt = time.time() - t0
    print(f"js_gcs: {len(jobs) - len(failed)} file(s), {fmt_size(done)} in {dt:.0f} s"
          f" ({fmt_size(done / max(dt, 1e-3))}/s)" + (f", {len(failed)} FAILED" if failed
                                                     else ""))
    return failed


def _summary(kinds_of, sizes, what):
    by = {}
    for p, k in kinds_of.items():
        if k in what:
            n, s = by.get(k, (0, 0))
            by[k] = (n + 1, s + sizes[p])
    return ", ".join(f"{k} {n} file(s) {fmt_size(s)}" for k, (n, s) in sorted(by.items()))


WILDCARDS = "*?["


def object_name(text, gcs):
    """'gs://BUCKET/x/y' or 'x/y' -> 'x/y' (exits for another bucket)."""
    t = text.strip()
    if t.startswith("gs://"):
        b, _, t = t[5:].partition("/")
        if b != gcs.bucket_name:
            sys.exit(f"js_gcs: {text} is not in bucket gs://{gcs.bucket_name} (use --bucket)")
    return t.strip("/")


def match(pattern, gcs):
    """{object name: (size, crc32c)} matching a pattern (* ? [..]; * also matches /)."""
    import fnmatch
    base = pattern[:min(pattern.find(c) for c in WILDCARDS if c in pattern)]
    return {n: v for n, v in gcs.list(base, directory=False).items()
            if fnmatch.fnmatchcase(n, pattern)}


def _finish(jobs, skipped, a, verb, where):
    """Print the plan; transfer unless --dry-run.  jobs: {name: (size, callable)}."""
    print(f"js_gcs: {len(jobs)} to {verb} ({fmt_size(sum(s for s, _ in jobs.values()))}), "
          f"{skipped} already {where}")
    if a.dry_run:
        for name, (size, _) in jobs.items():
            print(f"  would {verb}  {fmt_size(size):>9}  {name}")
        return 0
    todo = [(size, name, fn) for name, (size, fn) in jobs.items()]
    return 1 if _run(todo, a.jobs, "up  " if verb == "upload" else "down") else 0


def cmd_upload(a, gcs):
    jobs, skipped, incomplete = {}, 0, []
    prefix = a.prefix.strip("/")

    def consider(path, name, remote):
        nonlocal skipped
        if path.name.endswith(".h5") and not a.include_incomplete and h5_incomplete(path):
            incomplete.append(str(path))
        elif not a.force and same(path, remote, a.size_only):
            skipped += 1
        else:
            jobs[f"gs://{gcs.bucket_name}/{name}"] = (
                path.stat().st_size, lambda p=path, n=name: gcs.upload(p, n))

    dirs, files = [], {}
    for p in a.paths:
        p = Path(p).expanduser().resolve()
        if p.is_dir():
            dirs.append(p)
        elif p.is_file():
            files.setdefault(p.parent, []).append(p.name)
        else:
            sys.exit(f"js_gcs: {p} not found")
    # directories: the files of the selected kinds
    for d in dirs:
        kinds = classify(sorted(p.name for p in d.iterdir() if p.is_file()))
        dest = "/".join(x for x in (prefix, d.name) if x)
        remote = gcs.list(dest)
        sizes = {f: (d / f).stat().st_size for f in kinds}
        print(f"js_gcs: {d} -> gs://{gcs.bucket_name}/{dest}/: "
              f"{_summary(kinds, sizes, a.what) or 'nothing selected'}")
        for f, k in kinds.items():
            if k in a.what:
                consider(d / f, f"{dest}/{f}", remote.get(f"{dest}/{f}"))
    # files: each one named, whatever its kind; where its directory's upload puts it
    for parent, names in files.items():
        kinds = classify(sorted(p.name for p in parent.iterdir() if p.is_file()))
        dest = "/".join(x for x in (prefix, None if a.flat else parent.name) if x)
        for f in names:
            if kinds.get(f) is None:
                print(f"js_gcs: {parent / f} left out (temporary or hidden file)")
                continue
            name = f"{dest}/{f}" if dest else f
            print(f"js_gcs: {parent / f} ({kinds[f]}) -> gs://{gcs.bucket_name}/{name}")
            consider(parent / f, name, gcs.stat(name))
    if incomplete:
        print(f"js_gcs: {len(incomplete)} incomplete HDF5 file(s) left out "
              f"(--include-incomplete to upload them): " + ", ".join(
                  os.path.basename(p) for p in incomplete[:5])
              + (" ..." if len(incomplete) > 5 else ""))
    return _finish(jobs, skipped, a, "upload", "there")


def cmd_download(a, gcs):
    to = Path(a.to).expanduser().resolve()
    want = {}                                          # object name -> (size, crc), path
    for target in a.targets:
        t = object_name(target, gcs)
        if not t:
            sys.exit("js_gcs: download needs a directory, file or pattern in the bucket, "
                     "not the whole bucket")
        if any(c in t for c in WILDCARDS):             # a pattern: files, to TO/<file>
            found = match(t, gcs)
            kinds = classify(list(found))
            sel = {n: v for n, v in found.items() if kinds.get(n) in a.what}
            print(f"js_gcs: gs://{gcs.bucket_name}/{t} -> {to}/: {len(sel)} of "
                  f"{len(found)} matching file(s) selected")
            for n, v in sel.items():
                want[n] = (v, to / os.path.basename(n))
        elif (v := gcs.stat(t)) is not None:           # one object, to TO/<file>
            print(f"js_gcs: gs://{gcs.bucket_name}/{t} -> {to}/")
            want[t] = (v, to / os.path.basename(t))
        else:                                          # a directory, to TO/<last part>/
            remote = gcs.list(t)
            if not remote:
                sys.exit(f"js_gcs: nothing at gs://{gcs.bucket_name}/{t}")
            base = os.path.dirname(t)                  # keep the last part: .../NAME/x -> NAME/x
            rel = {n: os.path.relpath(n, base) if base else n for n in remote}
            kinds = classify(list(rel.values()))
            sizes = {rel[n]: remote[n][0] for n in remote}
            print(f"js_gcs: gs://{gcs.bucket_name}/{t}/ -> {to / os.path.basename(t)}/: "
                  f"{_summary(kinds, sizes, a.what) or 'nothing selected'}")
            for n in sorted(remote):
                if kinds.get(rel[n]) in a.what:
                    want[n] = (remote[n], to / rel[n])
    by_path = {}
    for n, (_, path) in want.items():
        if by_path.setdefault(path, n) != n:
            sys.exit(f"js_gcs: {by_path[path]} and {n} would both be {path}: download them "
                     "separately (--to)")
    jobs, skipped = {}, 0
    for n, (v, path) in sorted(want.items()):
        if not a.force and same(path, v, a.size_only):
            skipped += 1
        else:
            jobs[n] = (v[0], lambda n=n, p=path: gcs.download(n, p))
    return _finish(jobs, skipped, a, "download", "here")


def cmd_ls(a, gcs):
    prefix = object_name(a.prefix, gcs) if a.prefix else ""
    if any(c in prefix for c in WILDCARDS):
        remote = match(prefix, gcs)
    elif not a.recursive and a.what == set(KINDS):
        dirs, files = gcs.top_level(prefix)
        for d in dirs:
            print(f"  {'':>9}  gs://{gcs.bucket_name}/{d}")
        for n, (s, _) in sorted(files.items()):
            print(f"  {fmt_size(s):>9}  gs://{gcs.bucket_name}/{n}")
        if not dirs and not files:
            print(f"js_gcs: nothing under gs://{gcs.bucket_name}/{prefix}")
        return 0
    else:
        remote = gcs.list(prefix)
    kinds = classify(list(remote))
    sizes = {n: s for n, (s, _) in remote.items()}
    for n in sorted(remote):
        if kinds.get(n) in a.what:
            print(f"  {fmt_size(sizes[n]):>9}  {kinds[n]:<5}  gs://{gcs.bucket_name}/{n}")
    print(f"js_gcs: {_summary(kinds, sizes, a.what) or 'nothing'}")
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--key-file", dest="key_file", default=None,
                   help=f"service-account JSON key (default: see above; {KEY_NAME})")
    p.add_argument("--bucket", default=os.environ.get("JS_GCS_BUCKET", DEFAULT_BUCKET),
                   type=bucket_name, help="bucket (default gs://test_fno, or $JS_GCS_BUCKET)")
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

    up = sub.add_parser("upload", help="directories or files -> "
                        "gs://BUCKET/[PREFIX/]<dir name>/<file>")
    up.add_argument("paths", nargs="+", metavar="PATH",
                    help="production directories (their files of the --what kinds) and "
                         "files (each one, whatever its kind)")
    up.add_argument("--prefix", default="", help="put everything below this prefix")
    up.add_argument("--flat", action="store_true",
                    help="files go to gs://BUCKET/[PREFIX/]<file>, without their directory's "
                         "name (directories are not affected)")
    up.add_argument("--include-incomplete", action="store_true", dest="include_incomplete",
                    help="also HDF5 files whose 'complete' attribute is False")
    transfer_opts(up)
    down = sub.add_parser("download", help="directories, files or patterns in the bucket "
                          "-> TO/")
    down.add_argument("targets", nargs="+", metavar="TARGET",
                      help="a directory (-> TO/<its name>/), a file or a pattern with * ? "
                           "[..] (-> TO/<file>); NAME or gs://BUCKET/NAME; quote patterns")
    down.add_argument("--to", default=".", help="local directory (default .)")
    transfer_opts(down)
    ls = sub.add_parser("ls", help="list the bucket, or a directory in it")
    ls.add_argument("prefix", nargs="?", default="", help="a directory, or a pattern")
    ls.add_argument("--what", type=parse_what, default=set(KINDS),
                    help="pair, h5, root or all (lists recursively, with the kinds)")
    ls.add_argument("-r", "--recursive", action="store_true", help="every file below")
    st = sub.add_parser("setup", help="make the environment (--reinstall: again; --remove: "
                        "delete it)")
    g = st.add_mutually_exclusive_group()
    g.add_argument("--reinstall", action="store_true", help="make it again")
    g.add_argument("--remove", action="store_true", help="delete it")
    a = p.parse_args(argv)
    if getattr(a, "jobs", 1) < 1:
        p.error("-j must be >= 1")
    return a


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    a = parse_args(argv)
    if a.cmd == "setup":
        print(f"js_gcs: environment {ENV_DIR} ({sys.executable}): "
              + ", ".join(REQUIREMENTS))
        return 0
    try:
        import google.cloud.storage  # noqa: F401
        import google_crc32c  # noqa: F401
    except ImportError as e:
        sys.exit(f"js_gcs: {e}: the environment {ENV_DIR} is broken; "
                 "run ./js_gcs.py setup --reinstall")
    gcs = Gcs(a.bucket, find_key(a.key_file))
    return {"upload": cmd_upload, "download": cmd_download, "ls": cmd_ls}[a.cmd](a, gcs)


if __name__ == "__main__":
    if sys.argv[1:2] == ["setup"] and "--remove" in sys.argv[2:]:
        sys.exit(remove_env())                         # before (not inside) the environment
    if not _in_env() and os.environ.get("JS_GCS_NO_ENV") != "1":
        reexec_in_env(sys.argv[1:])
    sys.exit(main())
