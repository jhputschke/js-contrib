#!/usr/bin/env python3
"""upload_follow.py: upload a production's finished jobs to the OSDF (osdf:///fno4hic) while it
runs, and optionally delete the uploaded HDF5 files locally.

    ./upload_follow.py OUTBASE REMOTE [options]

OUTBASE is the production's output root on the host (launch_2gpu.sh: OUTBASE/gpu0, OUTBASE/gpu1;
any subdirectory holding run_jobs.sh output is taken).  Each subdirectory SUB goes to
REMOTE/SUB in the namespace; OUTBASE/seeds_used.tsv to REMOTE/seeds_used.tsv.

A job is uploaded once its <stem>.json says it is complete (the test run_jobs.sh uses to skip
it) and its files have not changed for --settle seconds: <stem>.h5 (pair), <stem>_particlize.h5,
<stem>.json, .log, .xml.  The upload is the tool's own (remote_transfer/js_osdf.py: the size the
origin stores is checked, the CRC32C goes into the remote manifest).

Deleting (--delete, default none) removes only HDF5 files that are VERIFIED: re-checked right
before deletion, the origin holds a file of the local size and the remote manifest has the CRC32C
of the local file.  --verify-every N also downloads every Nth file to be deleted and compares it
byte by byte first.  The .json/.log/.xml always stay: run_jobs.sh needs the .json to skip
finished jobs on a re-run.

    --delete none     keep everything (default)
    --delete pair     delete <stem>.h5 (the hydro history); keep <stem>_particlize.h5
    --delete all      delete <stem>.h5 and <stem>_particlize.h5
    --keep-free SIZE  with pair/all: delete only while the free space of OUTBASE's disk is below
                      SIZE (e.g. 200G), the oldest uploads first; otherwise right after upload

Modes: one pass (default), or --follow: a pass every --interval s until OUTBASE/.upload_final
exists or the process --follow-pid has ended, then a last pass that also uploads the
directories' other files (run_jobs.campaign, run_jobs.finished) and seeds_used.tsv.

State: OUTBASE/upload_state.json (what was uploaded, verified, deleted).  Needs
./remote_transfer/js_osdf.py login once (or --token-file).  Runs in js_osdf's environment.
"""
import argparse
import filecmp
import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

TOOL_DIR = Path(os.environ.get("JS_TRANSFER_DIR",
                               Path(__file__).resolve().parent / "remote_transfer"))
sys.path.insert(0, str(TOOL_DIR))
try:
    import js_osdf                                             # noqa: E402
except ImportError as e:
    sys.exit(f"upload_follow: remote_transfer not found in {TOOL_DIR} ({e}); "
             "set JS_TRANSFER_DIR")
core = js_osdf.core

if __name__ == "__main__":
    core.enter_env(js_osdf.ENV, __file__, sys.argv[1:], "JS_OSDF_NO_ENV")

PROG = "upload_follow"
core.PROG = PROG
STATE = "upload_state.json"
FINAL = ".upload_final"
VERIFY_DIR = ".upload_verify"
SMALL = (".json", ".log", ".xml")
DELETE_KINDS = {"none": (), "pair": ("pair",), "all": ("pair", "particlize")}
DIR_EXTRAS = ("run_jobs.campaign", "run_jobs.finished")


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def kind(p):
    """pair (<stem>.h5), particlize (<stem>_particlize.h5) or other."""
    if p.name.endswith("_particlize.h5"):
        return "particlize"
    return "pair" if p.suffix == ".h5" else "other"


def parse_size(text):
    m = re.fullmatch(r"\s*([\d.]+)\s*([kKmMgGtT]?)[bB]?\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"not a size: {text!r} (e.g. 200G, 1.5T)")
    return int(float(m.group(1)) * 1000 ** " kmgt".index(m.group(2).lower() or " "))


def fmt(n):
    return core.fmt_size(n)


class Follower:
    def __init__(self, a):
        self.a = a
        self.base = Path(a.outbase).expanduser().resolve()
        if not self.base.is_dir():
            sys.exit(f"{PROG}: {self.base} is not a directory")
        self.remote = a.remote.strip("/")
        self.delete_kinds = DELETE_KINDS[a.delete]
        self.state_path = self.base / STATE
        self.state = self._load_state()
        self.n_deleted_total = sum(1 for e in self.state.values() if e.get("deleted"))
        self.deletes_since_verify = 0
        self.deletion_blocked = None                 # a reason, after a failed verification
        cred = js_osdf.credential(SimpleNamespace(
            web=False, token_file=a.token_file, namespace=a.namespace, federation=a.federation))
        if cred is None:
            sys.exit(f"{PROG}: no credentials to write to {a.namespace}: "
                     f"{TOOL_DIR}/js_osdf.py login (or --token-file)")
        self.store = js_osdf.OsdfStore(a.namespace, a.federation, cred)

    # ── state ──────────────────────────────────────────────────────────────────
    def _load_state(self):
        try:
            return json.loads(self.state_path.read_text())["files"]
        except FileNotFoundError:
            return {}
        except (ValueError, KeyError) as e:
            sys.exit(f"{PROG}: {self.state_path} is not readable ({e}); move it away to start over"
                     " (files are then compared with the remote again)")

    def _save_state(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"remote": f"{self.store.label}/{self.remote}",
                                   "files": self.state}, indent=1, sort_keys=True))
        os.replace(tmp, self.state_path)

    # ── what to upload ───────────────────────────────────────────────────────────
    def subdirs(self):
        out = []
        for d in sorted(self.base.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and (
                    (d / "run_jobs.campaign").exists() or any(d.glob("*.json"))):
                out.append(d)
        return out

    def finished_jobs(self, d):
        """Stems of the jobs in d whose .json says complete."""
        stems = []
        for j in sorted(d.glob("*.json")):
            try:
                r = json.loads(j.read_text())
                ev = r["events_requested"]
                if r["events_written"] == ev and r.get("particlize_events_written", ev) == ev:
                    stems.append(j.name[:-5])
            except (ValueError, KeyError, OSError):
                continue                              # not a job summary, or being written
        return stems

    def candidates(self, final):
        """{local path: remote name} of files to upload now (new or changed since uploaded)."""
        now, out = time.time(), {}
        for d in self.subdirs():
            names = []
            for stem in self.finished_jobs(d):
                names += [stem + s for s in (".h5", "_particlize.h5") + SMALL]
            if final:
                names += list(DIR_EXTRAS)
            for n in names:
                p = d / n
                out[p] = f"{self.remote}/{d.name}/{n}"
        if final:
            p = self.base / "seeds_used.tsv"
            out[p] = f"{self.remote}/seeds_used.tsv"
        todo = {}
        for p, name in out.items():
            try:
                st = p.stat()
            except FileNotFoundError:
                continue                              # not written (or deleted after upload)
            if now - st.st_mtime < self.a.settle and not final:
                continue                              # still changing (log flushed at exit)
            e = self.state.get(str(p))
            if e and e["size"] == st.st_size and e["mtime_ns"] == st.st_mtime_ns:
                continue                              # uploaded, unchanged
            if p.suffix == ".h5" and core.h5_incomplete(p):
                continue
            todo[p] = name
        return todo

    # ── upload and verify ────────────────────────────────────────────────────────
    def upload(self, todo):
        if not todo:
            return 0, 0
        stats = {p: p.stat() for p in todo}
        jobs = {name: (stats[p].st_size, (lambda p=p, n=name: core.send(self.store, p, n)))
                for p, name in todo.items()}
        log(f"uploading {len(jobs)} file(s), {fmt(sum(s.st_size for s in stats.values()))}")
        failed, results, interrupted = core._run(jobs, self.a.jobs, "up  ")
        if results:
            core.write_manifests(self.store, results)
        ok = 0
        for p, name in todo.items():
            r = results.get(name)
            if r is None:
                continue
            st = stats[p]
            self.state[str(p)] = {"remote": name, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                                  "crc32c": r["crc32c"], "uploaded": time.time(),
                                  "deleted": None}
            ok += 1
        self._save_state()
        if interrupted:
            raise KeyboardInterrupt
        return ok, len(failed)

    def verified(self, p, e, manifests):
        """The origin holds p as uploaded: its size, and the manifest's CRC32C is p's."""
        st = self.store.stat(e["remote"])
        if st is None or st[0] != e["size"] or p.stat().st_size != e["size"]:
            return f"remote size {st and st[0]} vs local {p.stat().st_size}"
        d = os.path.dirname(e["remote"])
        if d not in manifests:
            manifests[d] = core.read_manifest(self.store, d)
        m = manifests[d].get(os.path.basename(e["remote"]))
        if not m or m.get("size") != e["size"] or m.get("crc32c") != e["crc32c"]:
            return "not in the remote manifest with this size and CRC32C"
        if core.crc32c_b64(p) != e["crc32c"]:
            return "local file changed since its upload (CRC32C)"
        return None

    def download_matches(self, p, e):
        tmp = self.base / VERIFY_DIR / p.name
        try:
            core.fetch(self.store, e["remote"], tmp, (e["size"], e["crc32c"]))
            return filecmp.cmp(tmp, p, shallow=False)
        finally:
            tmp.unlink(missing_ok=True)

    # ── delete ───────────────────────────────────────────────────────────────────
    def free(self):
        return shutil.disk_usage(self.base).free

    def delete(self):
        if not self.delete_kinds or self.deletion_blocked:
            return 0, 0, 0
        cands = []
        for k, e in self.state.items():
            p = Path(k)
            if not e.get("deleted") and kind(p) in self.delete_kinds and p.exists():
                cands.append((e["uploaded"], p, e))
        cands.sort(key=lambda c: c[0])                # oldest uploads first
        n, freed, requeued, manifests = 0, 0, 0, {}
        for _, p, e in cands:
            if self.a.keep_free is not None and self.free() >= self.a.keep_free:
                break
            why = self.verified(p, e, manifests)
            if why:
                log(f"NOT deleting {p}: {why}; uploading it again")
                self.state.pop(str(p))                # forget it: uploaded again
                requeued += 1
                continue
            self.deletes_since_verify += 1
            if self.a.verify_every and self.deletes_since_verify >= self.a.verify_every:
                self.deletes_since_verify = 0
                if not self.download_matches(p, e):
                    self.deletion_blocked = f"downloaded {e['remote']} differs from {p}"
                    log(f"ERROR: {self.deletion_blocked}: no more deletions in this run")
                    break
                log(f"verified by download: {e['remote']}")
            size = p.stat().st_size
            p.unlink()
            e["deleted"] = time.time()
            n, freed = n + 1, freed + size
        self.n_deleted_total += n
        if n or requeued:
            self._save_state()
        return n, freed, requeued

    # ── passes ───────────────────────────────────────────────────────────────────
    def one_pass(self, final=False):
        t0 = time.time()
        todo = self.candidates(final)
        nbytes = sum(p.stat().st_size for p in todo)
        ok, failed = self.upload(todo)
        dt = time.time() - t0
        n_del, freed, requeued = self.delete()
        if requeued:                                  # failed verification: upload again now
            again = self.candidates(final=True)
            nbytes += sum(p.stat().st_size for p in again)
            ok2, failed2 = self.upload(again)
            ok, failed = ok + ok2, failed + failed2
            n2, freed2, _ = self.delete()
            n_del, freed = n_del + n2, freed + freed2
            dt = time.time() - t0
        up = list(self.state.values())
        msg = (f"pass{' (final)' if final else ''}: {ok} uploaded"
               + (f" ({fmt(nbytes)}, {fmt(nbytes / max(dt, 1e-3))}/s)" if ok else "")
               + (f", {failed} FAILED (retried next pass)" if failed else "")
               + (f", {n_del} deleted ({fmt(freed)})" if n_del else "")
               + f"; total {len(up)} files {fmt(sum(e['size'] for e in up))} uploaded,"
               f" {self.n_deleted_total} deleted locally; free {fmt(self.free())}")
        log(msg)
        if self.free() < self.a.warn_free:
            log(f"WARNING: only {fmt(self.free())} free on {self.base} "
                f"(--warn-free {fmt(self.a.warn_free)})")
        return failed

    def stop_requested(self):
        if (self.base / FINAL).exists():
            return True
        if self.a.follow_pid:
            try:
                os.kill(self.a.follow_pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
        return False

    def run(self):
        log(f"{self.base} -> {self.store.label}/{self.remote}/  delete={self.a.delete}"
            + (f" keep-free={fmt(self.a.keep_free)}" if self.a.keep_free is not None else "")
            + (f" verify-every={self.a.verify_every}" if self.a.delete != "none" else ""))
        if self.a.follow:
            while not self.stop_requested():
                self.one_pass()
                for _ in range(int(self.a.interval)):
                    if self.stop_requested():
                        break
                    time.sleep(1)
        failed = self.one_pass(final=True)
        (self.base / FINAL).unlink(missing_ok=True)
        try:
            (self.base / VERIFY_DIR).rmdir()
        except OSError:
            pass
        if failed:
            log(f"{failed} file(s) failed: run again ({sys.argv[0]} {self.a.outbase} "
                f"{self.a.remote} ...) to retry")
        return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outbase", help="the production's output root on the host")
    p.add_argument("remote", help="remote folder in the namespace, e.g. AuAu_pth3_c1")
    p.add_argument("--delete", choices=tuple(DELETE_KINDS), default="none")
    p.add_argument("--keep-free", type=parse_size, default=None, metavar="SIZE",
                   help="with --delete pair/all: delete only while free space < SIZE")
    p.add_argument("--verify-every", type=int, default=20, metavar="N",
                   help="download every Nth file before deleting it and compare (default 20; "
                        "0: never)")
    p.add_argument("--follow", action="store_true", help="repeat until .upload_final / --follow-pid")
    p.add_argument("--follow-pid", type=int, default=None, metavar="PID")
    p.add_argument("--interval", type=float, default=60, help="s between passes (default 60)")
    p.add_argument("--settle", type=float, default=30,
                   help="s a finished job's files must be unchanged (default 30)")
    p.add_argument("--warn-free", type=parse_size, default=parse_size("100G"), metavar="SIZE")
    p.add_argument("-j", "--jobs", type=int, default=4, help="files at once (default 4)")
    p.add_argument("--namespace", default=os.environ.get("JS_OSDF_NAMESPACE", js_osdf.DEFAULT_NAMESPACE))
    p.add_argument("--federation", default=js_osdf.OSDF)
    p.add_argument("--token-file", default=None)
    a = p.parse_args()
    if a.keep_free is not None and a.delete == "none":
        p.error("--keep-free needs --delete pair or --delete all")
    def on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    f = Follower(a)
    try:
        return f.run()
    except KeyboardInterrupt:
        f._save_state()
        log("interrupted: state saved; run again to finish")
        return 130


if __name__ == "__main__":
    sys.exit(main())
