"""
utils/remote_transfer/test/test_transfer.py

js_gcs.py and js_osdf.py without the network.  The commands of transfer_core run against
two stores: an in-memory stand-in for a GCS bucket (CRC32C from the store), and
js_osdf.OsdfStore on fsspec's in-memory file system in place of pelicanfs (sizes only, so
the CRC32C manifest).  Also: file kinds, --what, the key and token lookup, names, the
environment's removal, rm, and the browser login against a fake Pelican issuer.

    pytest utils/remote_transfer/test -q
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the scripts, one up
google_crc32c = pytest.importorskip("google_crc32c")
h5py = pytest.importorskip("h5py")
memory = pytest.importorskip("fsspec.implementations.memory")

import js_gcs  # noqa: E402
import js_osdf  # noqa: E402
import transfer_core as core  # noqa: E402

STEM = "AuAu_jet_0001"
PROD = {f"{STEM}.h5": "pair", f"{STEM}.json": "pair", f"{STEM}.xml": "pair",
        f"{STEM}.log": "pair", f"{STEM}_particlize.h5": "h5",
        f"{STEM}_hadrons_bulk_jet.h5": "h5", f"{STEM}_hadrons_bulk_bg.h5": "h5",
        f"{STEM}_hadrons_jet_frag.h5": "h5", f"{STEM}_hadronize.log": "h5",
        f"{STEM}_hadrons.root": "root", "pth10-40_campaign.root": "root",
        "wake_observables.h5": "other", "run_jobs.campaign": "other"}


def _crc(b):
    return base64.b64encode(google_crc32c.Checksum(b).digest()).decode()


class FakeGcs:
    """A GCS bucket on a dict (name -> bytes), with the store's own CRC32C."""

    native_checksums, manifest, label = True, False, "gs://test_fno"

    def __init__(self):
        self.objects, self.uploads, self.downloads = {}, [], []
        self.relative = js_gcs.GcsStore("test_fno", "k").relative

    def list(self, prefix):
        p = prefix.strip("/") + "/" if prefix.strip("/") else ""
        return {n: (len(b), _crc(b)) for n, b in self.objects.items() if n.startswith(p)}

    def stat(self, name):
        b = self.objects.get(name)
        return None if b is None else (len(b), _crc(b))

    def upload(self, path, name):
        self.uploads.append(name)
        self.objects[name] = Path(path).read_bytes()

    def download(self, name, path):
        self.downloads.append(name)
        Path(path).write_bytes(self.objects[name])

    def read_bytes(self, name):
        return self.objects.get(name)

    def delete(self, name):
        if self.objects.pop(name, None) is None:
            raise FileNotFoundError(name)

    def remove_dir(self, name):
        self.delete(name + "/")

    def top_level(self, prefix):
        p = prefix.strip("/") + "/" if prefix.strip("/") else ""
        below = [n[len(p):] for n in self.objects if n.startswith(p)]
        return (sorted({p + b.split("/")[0] + "/" for b in below if "/" in b}),
                {p + b: self.stat(p + b) for b in below if "/" not in b})

    def data(self):
        return dict(self.objects)


class MemOsdf(js_osdf.OsdfStore):
    """OsdfStore on fsspec's memory file system, counting the transfers."""

    def __init__(self):
        fs = memory.MemoryFileSystem()
        fs.store.clear()
        fs.pseudo_dirs[:] = [""]
        super().__init__("/fno4hic", fs=fs)
        self.uploads, self.downloads = [], []
        # the memory file system shares one file object (and its position) between readers:
        # one download at a time (pelicanfs's are separate HTTP requests)
        self._lock = threading.Lock()

    def upload(self, path, name):
        if not name.endswith(core.MANIFEST):
            self.uploads.append(name)
        super().upload(path, name)

    def download(self, name, path):
        with self._lock:
            self.downloads.append(name)
            super().download(name, path)

    def _rm(self, name):                       # what the origin does with a DELETE
        p = self.path(name)
        if self.fs.isfile(p):
            self.fs.rm_file(p)
        elif not self.fs.isdir(p):
            raise FileNotFoundError(p)
        elif self.fs.ls(p):
            raise OSError(f"{p}/ is not empty")
        else:
            self.fs.rmdir(p)

    def data(self):
        return {self.name_of(p): self.fs.cat_file(p) for p in self.fs.find(self.ns)
                if not p.endswith(core.MANIFEST)}

    def manifest_of(self, d):
        return json.loads(self.fs.cat_file(f"{self.ns}/{d}/{core.MANIFEST}"))["files"]


@pytest.fixture(params=["gcs", "osdf"])
def store(request):
    return FakeGcs() if request.param == "gcs" else MemOsdf()


def _args(store, argv):
    mod = js_gcs if isinstance(store, FakeGcs) else js_osdf
    return mod.parse_args(argv)


def _up(store, argv):
    return core.cmd_upload(_args(store, ["upload", *argv]), store)


def _down(store, argv):
    return core.cmd_download(_args(store, ["download", *argv]), store)


def _prod(tmp_path, incomplete=()):
    d = tmp_path / "AuAu_a"
    d.mkdir()
    for name in PROD:
        if name.endswith(".h5"):
            with h5py.File(d / name, "w") as f:
                f.attrs["complete"] = name not in incomplete
                f["x"] = [len(name)]
        else:
            (d / name).write_text(name * 10)
    (d / f"{STEM}_hadrons.root.part").write_text("being written")
    (d / "sub").mkdir()
    (d / "sub" / "x.root").write_text("not uploaded: a subdirectory")
    return d


# ── pieces ────────────────────────────────────────────────────────────────────────
def test_kinds_and_what():
    assert core.classify(list(PROD) + ["a.part", ".hidden", core.MANIFEST]) == PROD
    # remote listings: per directory
    assert core.classify([f"x/{n}" for n in PROD]) == {f"x/{n}": k for n, k in PROD.items()}
    assert core.classify(["x/A.h5", "y/A.json"]) == {"x/A.h5": "other", "y/A.json": "other"}
    assert core.parse_what("pair,root") == {"pair", "root"}
    assert core.parse_what("all") == set(core.KINDS)
    with pytest.raises(Exception, match="not pair, h5, root or all"):
        core.parse_what("hdf5")
    assert js_gcs.bucket_name("gs://test_fno/") == "test_fno"


def test_crc32c_as_gcs(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"hello world")
    assert core.crc32c_b64(p) == "yZRlqg=="         # GCS's crc32c of "hello world"
    assert core.same(p, (11, "yZRlqg=="), False)
    assert not core.same(p, (11, "AAAAAA=="), False)
    assert core.same(p, (11, "AAAAAA=="), True)     # --size-only
    assert core.same(p, (11, None), False)          # no checksum known: the size
    assert not core.same(p, (12, "yZRlqg=="), True)


def test_find_key(tmp_path, monkeypatch):
    for v in ("JS_GCS_KEY_FILE", "GOOGLE_APPLICATION_CREDENTIALS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(js_gcs, "__file__", str(tmp_path / "bin" / "js_gcs.py"))
    with pytest.raises(SystemExit, match="no key file"):
        js_gcs.find_key(None)
    (tmp_path / js_gcs.KEY_NAME).write_text("{}")
    assert js_gcs.find_key(None) == tmp_path / js_gcs.KEY_NAME
    other = tmp_path / "k.json"
    other.write_text("{}")
    monkeypatch.setenv("JS_GCS_KEY_FILE", str(other))
    assert js_gcs.find_key(None) == other
    with pytest.raises(SystemExit, match="not found"):
        js_gcs.find_key(str(tmp_path / "missing.json"))


def test_find_token(tmp_path, monkeypatch):
    for v in ("JS_OSDF_TOKEN_FILE", "BEARER_TOKEN_FILE", "BEARER_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(js_osdf, "__file__", str(tmp_path / "bin" / "js_osdf.py"))
    assert js_osdf.find_token(None) is None          # pelicanfs then looks for one itself
    cfg = tmp_path / "home" / ".config" / "js_osdf"
    cfg.mkdir(parents=True)
    (cfg / js_osdf.TOKEN_NAME).write_text("from-config\n")
    assert js_osdf.find_token(None) == "from-config"
    monkeypatch.setenv("BEARER_TOKEN", "from-env")
    assert js_osdf.find_token(None) == "from-env"
    f = tmp_path / "t"
    f.write_text("from-file")
    monkeypatch.setenv("BEARER_TOKEN_FILE", str(f))
    assert js_osdf.find_token(None) == "from-file"
    with pytest.raises(SystemExit, match="not found"):
        js_osdf.find_token(str(tmp_path / "missing"))


def test_osdf_names():
    s = MemOsdf()
    assert s.label == "osdf:///fno4hic"
    for text in ("osdf:///fno4hic/a/b", "/fno4hic/a/b", "a/b", "a/b/",
                 "pelican://osg-htc.org/fno4hic/a/b"):
        assert s.relative(text) == "a/b"
    assert s.relative("osdf:///fno4hic") == ""
    with pytest.raises(SystemExit, match="not in osdf:///fno4hic"):
        s.relative("osdf:///other/a")
    assert js_osdf.namespace_path("fno4hic/") == "/fno4hic"
    other = js_osdf.OsdfStore("/ns", federation="fed.example.org", fs=s.fs)
    assert other.label == "pelican://fed.example.org/ns"


def test_remove_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JS_X_ENV", str(tmp_path / "js_x" / "venv"))
    env = core.Env("js_x", ("pkg",), "JS_X_ENV")
    assert env.remove() == 0 and "no environment" in capsys.readouterr().out
    env.dir.mkdir(parents=True)
    (env.dir / "keep.txt").write_text("not a venv")
    assert env.remove() == 1 and (env.dir / "keep.txt").exists()     # left alone
    (env.dir / "pyvenv.cfg").write_text("")
    env.stamp.write_text("pkg")
    assert env.remove() == 0
    assert not env.dir.exists() and not env.dir.parent.exists()


# ── commands, on both stores ──────────────────────────────────────────────────────
@pytest.mark.parametrize("what,kinds", [("pair", {"pair"}), ("h5", {"h5"}),
                                        ("root", {"root"}), ("pair,root", {"pair", "root"}),
                                        ("all", {"pair", "h5", "root", "other"})])
def test_upload_what(tmp_path, store, what, kinds):
    d = _prod(tmp_path)
    assert _up(store, [str(d), "--what", what, "--prefix", "p"]) == 0
    data = store.data()
    assert sorted(data) == sorted(f"p/AuAu_a/{n}" for n, k in PROD.items() if k in kinds)
    for name, b in data.items():
        assert b == (d / name.rsplit("/", 1)[1]).read_bytes()


def test_upload_skip_incomplete_force(tmp_path, store, capsys):
    d = _prod(tmp_path, incomplete=(f"{STEM}_hadrons_jet_frag.h5",))
    assert _up(store, [str(d), "--what", "h5", "--dry-run"]) == 0
    assert store.uploads == [] and "would upload" in capsys.readouterr().out
    assert _up(store, [str(d), "--what", "h5"]) == 0
    out = capsys.readouterr().out
    assert "1 incomplete HDF5 file(s) left out" in out and "4 to upload" in out
    assert f"AuAu_a/{STEM}_hadrons_jet_frag.h5" not in store.data()
    # again: everything there is skipped; a changed file of the same size goes up again
    # (GCS: its CRC32C; OSDF: the manifest's)
    store.uploads.clear()
    p = d / f"{STEM}_hadronize.log"
    p.write_text(p.read_text().upper())
    assert _up(store, [str(d), "--what", "h5"]) == 0
    assert store.uploads == [f"AuAu_a/{STEM}_hadronize.log"]
    assert "3 already there" in capsys.readouterr().out
    store.uploads.clear()
    p.write_text(p.read_text().lower())              # --size-only: not seen
    _up(store, [str(d), "--what", "h5", "--size-only"])
    assert store.uploads == []
    _up(store, [str(d), "--what", "h5", "--force", "--include-incomplete"])
    assert len(store.uploads) == 5


def test_upload_files(tmp_path, store, capsys):
    d = _prod(tmp_path, incomplete=(f"{STEM}_hadrons_jet_frag.h5",))
    # named files, whatever --what says; next to where their directory goes, or --flat
    assert _up(store, [str(d / f"{STEM}.json"), str(d / "run_jobs.campaign"), "--what",
                       "root", "--prefix", "p"]) == 0
    assert sorted(store.data()) == [f"p/AuAu_a/{STEM}.json", "p/AuAu_a/run_jobs.campaign"]
    assert f"(pair) -> {store.label}/p/AuAu_a/{STEM}.json" in capsys.readouterr().out
    _up(store, [str(d / f"{STEM}.xml"), "--flat"])
    _up(store, [str(d / f"{STEM}.xml"), "--flat", "--prefix", "q"])
    assert {f"{STEM}.xml", f"q/{STEM}.xml"} <= set(store.data())
    # skipped when there, incomplete and .part left out, a directory and a file in it once
    store.uploads.clear()
    _up(store, [str(d / f"{STEM}.json"), str(d / f"{STEM}_hadrons_jet_frag.h5"),
                str(d / f"{STEM}_hadrons.root.part"), "--prefix", "p"])
    out = capsys.readouterr().out
    assert store.uploads == [] and "1 already there" in out and "1 incomplete" in out
    assert "left out (temporary or hidden file)" in out
    _up(store, [str(d), str(d / f"{STEM}.log"), "--what", "pair"])
    assert sorted(store.uploads) == sorted(f"AuAu_a/{n}" for n, k in PROD.items()
                                           if k == "pair")
    with pytest.raises(SystemExit, match="not found"):
        _up(store, [str(d / "missing.h5")])


def test_download(tmp_path, store):
    d = _prod(tmp_path)
    _up(store, [str(d), "--prefix", "prod"])
    to = tmp_path / "local"
    assert _down(store, [f"{store.label}/prod/AuAu_a", "--to", str(to), "--what",
                         "root"]) == 0
    got = sorted(p.name for p in (to / "AuAu_a").iterdir())
    assert got == sorted(n for n, k in PROD.items() if k == "root")
    assert not list(to.rglob("*.part"))
    store.downloads.clear()
    assert _down(store, ["prod/AuAu_a", "--to", str(to)]) == 0
    assert len(store.downloads) == len(PROD) - 2      # the root files are there already
    for n in PROD:
        assert (to / "AuAu_a" / n).read_bytes() == (d / n).read_bytes()
    assert not (to / "AuAu_a" / core.MANIFEST).exists()
    with pytest.raises(SystemExit, match="not in"):
        _down(store, ["gs://other/prod/AuAu_a" if isinstance(store, FakeGcs)
                      else "osdf:///other/prod/AuAu_a"])
    with pytest.raises(SystemExit, match="nothing at"):
        _down(store, ["prod/missing"])


def test_download_files_and_patterns(tmp_path, store, capsys):
    d = _prod(tmp_path)
    _up(store, [str(d), "--prefix", "prod"])
    to = tmp_path / "local"
    # one file, and a pattern (with --what) -> TO/<file>
    assert _down(store, [f"{store.label}/prod/AuAu_a/{STEM}.xml",
                         f"prod/AuAu_a/{STEM}_hadrons*", "--what", "h5", "--to",
                         str(to)]) == 0
    assert sorted(p.name for p in to.iterdir()) == sorted(
        [f"{STEM}.xml"] + [f"{STEM}_hadrons_{t}.h5" for t in core.HADRON_TAGS])
    assert (to / f"{STEM}.xml").read_bytes() == (d / f"{STEM}.xml").read_bytes()
    assert "3 of 4 matching file(s) selected" in capsys.readouterr().out   # not the .root
    store.downloads.clear()
    _down(store, [f"prod/AuAu_a/{STEM}.x?l", "--to", str(to)])
    assert store.downloads == [] and "1 already here" in capsys.readouterr().out
    # one file via a directory and a pattern: both places
    store.downloads.clear()
    _down(store, ["prod/AuAu_a", f"prod/AuAu_a/{STEM}.js*", "--what", "pair", "--to",
                  str(tmp_path / "both")])
    assert (tmp_path / "both" / f"{STEM}.json").exists()
    assert (tmp_path / "both" / "AuAu_a" / f"{STEM}.json").exists()
    assert store.downloads.count(f"prod/AuAu_a/{STEM}.json") == 2
    # two files to one local path: refused
    _up(store, [str(d / f"{STEM}.xml"), "--flat", "--prefix", "other"])
    with pytest.raises(SystemExit, match="would both be"):
        _down(store, [f"*/{STEM}.xml", "--to", str(to)])
    with pytest.raises(SystemExit, match="not the whole"):
        _down(store, [f"{store.label}/"])


# ── the manifest (OSDF) ───────────────────────────────────────────────────────────
def test_manifest(tmp_path, capsys):
    store = MemOsdf()
    d = _prod(tmp_path)
    assert _up(store, [str(d), "--what", "root"]) == 0
    m = store.manifest_of("AuAu_a")
    assert set(m) == {n for n, k in PROD.items() if k == "root"}
    for n, e in m.items():
        assert e["size"] == (d / n).stat().st_size and e["crc32c"] == core.crc32c_b64(d / n)
    # a later upload to the same directory adds to the manifest
    _up(store, [str(d / f"{STEM}.json")])
    assert f"{STEM}.json" in store.manifest_of("AuAu_a") and len(
        store.manifest_of("AuAu_a")) == 3
    # ls doesn't show it
    assert core.cmd_ls(js_osdf.parse_args(["ls", "AuAu_a"]), store) == 0
    assert core.MANIFEST not in capsys.readouterr().out
    # a file another tool put there (no manifest entry): compared by size
    store.fs.pipe_file("/fno4hic/AuAu_a/run_jobs.campaign", b"x" * (d / "run_jobs.campaign")
                       .stat().st_size)
    store.uploads.clear()
    _up(store, [str(d / "run_jobs.campaign")])
    assert store.uploads == []
    # a download whose bytes don't match the manifest fails, leaving nothing behind
    name = f"/fno4hic/AuAu_a/{STEM}_hadrons.root"
    store.fs.pipe_file(name, b"y" * len(store.fs.cat_file(name)))
    to = tmp_path / "local"
    assert _down(store, [f"AuAu_a/{STEM}_hadrons.root", "--to", str(to)]) == 1
    assert "CRC32C differs" in capsys.readouterr().out
    assert not list(to.iterdir())
    # the directory download fails the same way
    assert _down(store, ["AuAu_a", "--what", "root", "--to", str(to)]) == 1
    # an unreadable manifest: a warning, sizes only
    store.fs.pipe_file(f"/fno4hic/AuAu_a/{core.MANIFEST}", b"not json")
    store.uploads.clear()
    _up(store, [str(d), "--what", "pair"])
    assert "is not readable: ignored" in capsys.readouterr().out


def test_upload_as(tmp_path, store):
    d = _prod(tmp_path)
    assert _up(store, [str(d), "--what", "root", "--prefix", "p", "--as", "AuAu_new"]) == 0
    assert sorted(store.data()) == sorted(f"p/AuAu_new/{n}" for n, k in PROD.items()
                                          if k == "root")
    # files of that directory join it; again: skipped
    _up(store, [str(d / f"{STEM}.json"), "--as", "nested/AuAu_new/"])
    assert f"nested/AuAu_new/{STEM}.json" in store.data()
    store.uploads.clear()
    _up(store, [str(d), str(d / f"{STEM}.json"), "--what", "root", "--prefix", "p",
                "--as", "AuAu_new"])
    assert store.uploads == [f"p/AuAu_new/{STEM}.json"]
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.txt").write_text("x")
    with pytest.raises(SystemExit, match="one directory only"):
        _up(store, [str(d), str(other), "--as", "x"])
    with pytest.raises(SystemExit, match="don't go together"):
        _up(store, [str(d / f"{STEM}.json"), "--as", "x", "--flat"])
    with pytest.raises(SystemExit, match="needs a name"):
        _up(store, [str(d), "--as", "/"])


# ── the browser login (js_osdf.WebLogin) against a fake issuer ────────────────────────
def _jwt(**claims):
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"e30.{body}.sig"


class FakeIssuer:
    """The OSDF's pelican-configuration, its director and a Pelican token issuer, in place
    of js_osdf._http.  refuse_refresh: the refresh token is no longer accepted."""

    ISS = "https://origin.example.org:8455"

    def __init__(self):
        self.calls, self.pending, self.n, self.refuse_refresh = [], 2, 0, False

    def token(self):
        self.n += 1
        return {"access_token": _jwt(exp=int(js_osdf.time.time()) + 1200, sub="user@wayne",
                                     jti=self.n, scope="storage.read:/ storage.create:/"),
                "refresh_token": f"refresh-{self.n}", "expires_in": 1200}

    def __call__(self, url, form=None, json_body=None, redirect=True, timeout=30):
        self.calls.append((url, form or json_body))
        if url == "https://osg-htc.org/.well-known/pelican-configuration":
            return 200, {}, b'{"director_endpoint": "https://director.example.org"}'
        if url == "https://director.example.org/api/v1.0/director/origin/fno4hic/":
            assert not redirect
            return 307, {"X-Pelican-Token-Generation": f"issuer={self.ISS}, max-scope-depth=3,"
                         " strategy=OAuth2, base-path=/fno4hic"}, b""
        if url == f"{self.ISS}/.well-known/openid-configuration":
            return 200, {}, json.dumps({k: f"{self.ISS}/{k}" for k in (
                "token_endpoint", "device_authorization_endpoint", "registration_endpoint",
                "revocation_endpoint")}).encode()
        if url == f"{self.ISS}/registration_endpoint":
            assert "storage.create:/" in json_body["scope"]
            return 201, {}, b'{"client_id": "cid", "client_secret": "sec"}'
        if url == f"{self.ISS}/device_authorization_endpoint":
            assert form["client_id"] == "cid" and "storage.modify:/" in form["scope"]
            return 200, {}, json.dumps({
                "device_code": "dc", "user_code": "ABC", "interval": 5, "expires_in": 900,
                "verification_uri": f"{self.ISS}/device",
                "verification_uri_complete": f"{self.ISS}/device?user_code=ABC"}).encode()
        if url == f"{self.ISS}/token_endpoint":
            if form["grant_type"] == js_osdf.DEVICE_GRANT:
                assert "storage.create:/" in form["scope"]
                if self.pending:                       # Pelican: pending, then consent
                    self.pending -= 1
                    return 400, {}, json.dumps({"error": (
                        "authorization_pending" if self.pending else "consent_required")}).encode()
                return 200, {}, json.dumps(self.token()).encode()
            assert form["grant_type"] == "refresh_token"
            if self.refuse_refresh:
                return 400, {}, b'{"error": "invalid_grant"}'
            return 200, {}, json.dumps(self.token()).encode()
        if url == f"{self.ISS}/revocation_endpoint":
            return 200, {}, b""
        raise AssertionError(url)


def test_web_login(tmp_path, monkeypatch, capsys):
    issuer = FakeIssuer()
    monkeypatch.setattr(js_osdf, "_http", issuer)
    monkeypatch.setattr(js_osdf, "WEB_DIR", str(tmp_path / "web"))
    monkeypatch.setattr(js_osdf.time, "sleep", lambda s: None)
    for v in ("JS_OSDF_TOKEN_FILE", "BEARER_TOKEN_FILE", "BEARER_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(js_osdf, "__file__", str(tmp_path / "bin" / "js_osdf.py"))
    assert js_osdf.credential(js_osdf.parse_args(["ls"])) is None      # no login yet

    assert js_osdf.cmd_login(js_osdf.parse_args(["login"])) == 0
    out = capsys.readouterr()
    assert f"{issuer.ISS}/device?user_code=ABC" in out.err and "user@wayne" in out.out
    w = js_osdf.WebLogin("/fno4hic")
    assert w.file.stat().st_mode & 0o777 == 0o600 and w.state["refresh_token"] == "refresh-1"

    # later commands use it without --web; the store asks for the token before each request
    cred = js_osdf.credential(js_osdf.parse_args(["upload", "x"]))
    fs = memory.MemoryFileSystem()
    s = js_osdf.OsdfStore("/fno4hic", token=cred, fs=fs)
    assert s._headers["Authorization"] == f"Bearer {w.state['access_token']}"
    # expired: renewed from the refresh token, the next request carries the new one
    st = json.loads(w.file.read_text())
    st["expires_at"] = 0
    w.file.write_text(json.dumps(st))
    cred = js_osdf.credential(js_osdf.parse_args(["ls"]))
    s = js_osdf.OsdfStore("/fno4hic", token=cred, fs=fs)
    assert json.loads(w.file.read_text())["refresh_token"] == "refresh-2"
    first = s._headers["Authorization"]
    cred.login.state["expires_at"] = 0           # expires mid-run
    cred.login._save()
    s.list("")
    assert s._headers["Authorization"] != first and fs.token == s._headers["Authorization"]
    # the refresh token refused: a run already going fails its requests, a new one goes on
    # without a token (reads of a public namespace), and --web logs in again
    issuer.refuse_refresh = True
    cred.login.state["expires_at"] = 0
    cred.login._save()
    with pytest.raises(RuntimeError, match="expired"):
        s.upload(tmp_path / "f", "f")
    assert js_osdf.credential(js_osdf.parse_args(["ls"])) is None
    assert "has expired" in capsys.readouterr().out
    issuer.pending = 0
    assert callable(js_osdf.credential(js_osdf.parse_args(["ls", "--web"])))
    with pytest.raises(SystemExit, match="not both"):
        js_osdf.credential(js_osdf.parse_args(["--web", "--token-file", "t", "ls"]))

    assert js_osdf.cmd_logout(js_osdf.parse_args(["logout"])) == 0
    assert not w.file.exists() and issuer.calls[-1][0].endswith("revocation_endpoint")
    assert "logged out" in capsys.readouterr().out


def _rm(store, argv):
    return core.cmd_rm(_args(store, ["rm", *argv]), store)


def test_rm(tmp_path, store, capsys):
    d = _prod(tmp_path)
    assert _up(store, [str(d)]) == 0
    _up(store, [str(d / f"{STEM}.xml"), "--as", "AuAu_a/sub/deeper"])
    every = set(store.data())
    with pytest.raises(SystemExit, match="not the whole"):
        _rm(store, ["", "--yes"])
    with pytest.raises(SystemExit, match="is a directory: rm -r"):
        _rm(store, ["AuAu_a", "--yes"])
    with pytest.raises(SystemExit, match="nothing at"):
        _rm(store, ["AuAu_b", "--yes"])
    assert _rm(store, ["AuAu_b", "-r", "--yes"]) == 1
    assert "nothing at" in capsys.readouterr().out
    assert _rm(store, ["-r", "AuAu_a", "--dry-run"]) == 0
    assert set(store.data()) == every and "would remove" in capsys.readouterr().out
    # one file and a pattern; their manifest entries go too
    assert _rm(store, [f"AuAu_a/{STEM}.log", f"{store.label}/AuAu_a/*_hadrons_bulk_*",
                       "--yes"]) == 0
    gone = {f"AuAu_a/{STEM}.log", f"AuAu_a/{STEM}_hadrons_bulk_jet.h5",
            f"AuAu_a/{STEM}_hadrons_bulk_bg.h5"}
    assert set(store.data()) == every - gone
    if isinstance(store, MemOsdf):
        assert not {os.path.basename(n) for n in gone} & set(store.manifest_of("AuAu_a"))
    # emptied by a pattern: an empty directory, which no file listing shows
    assert _rm(store, ["AuAu_a/sub/deeper/*", "--yes"]) == 0
    if isinstance(store, MemOsdf):
        assert store.fs.isdir("/fno4hic/AuAu_a/sub/deeper") and not store.list("AuAu_a/sub")
        assert _rm(store, ["-r", "AuAu_a/sub", "--yes"]) == 0       # only empty ones
        assert not store.fs.exists("/fno4hic/AuAu_a/sub")
    # a kind out of a directory: the directory stays
    assert _rm(store, ["-r", "AuAu_a", "--what", "root", "--yes"]) == 0
    assert not [n for n in store.data() if n.endswith(".root")] and store.data()
    # asks first: no terminal, no; then yes
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(SystemExit, match="--yes"):
        _rm(store, ["-r", "AuAu_a"])
    monkey.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkey.setattr("builtins.input", lambda q: "n")
    assert _rm(store, ["-r", "AuAu_a"]) == 1 and store.data()
    assert "nothing removed" in capsys.readouterr().out
    monkey.setattr("builtins.input", lambda q: "y")
    assert _rm(store, ["-r", "AuAu_a"]) == 0
    monkey.undo()
    # all of it: the files, the manifests and the directories, the deepest first
    assert store.data() == {} and store.list("") == {}
    if isinstance(store, MemOsdf):
        assert not store.fs.exists("/fno4hic/AuAu_a")


def test_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(js_osdf, "WEB_DIR", str(tmp_path / "web"))
    for v in ("JS_OSDF_TOKEN_FILE", "BEARER_TOKEN_FILE", "BEARER_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(js_osdf, "__file__", str(tmp_path / "bin" / "js_osdf.py"))
    status = lambda: js_osdf.cmd_status(js_osdf.parse_args(["status"]))   # noqa: E731
    assert status() == 1 and "no browser login" in capsys.readouterr().out
    now = js_osdf.time.time()

    def refresh(issued, lifetime_s):                  # an OA4MP refresh token
        url = (f"https://iss.example.org:8455/refreshToken/x/{int(issued * 1000)}?type="
               f"refreshToken&ts={int(issued * 1000)}&version=v2.0&lifetime={lifetime_s * 1000}")
        return base64.b32encode(url.encode()).decode().rstrip("=")
    assert abs(js_osdf._refresh_expiry(refresh(now, 3600)) - (now + 3600)) < 1
    assert js_osdf._refresh_expiry("not-a-token") is None
    w = js_osdf.WebLogin("/fno4hic")
    w.state = {"access_token": _jwt(exp=int(now) + 600), "expires_at": int(now) + 600,
               "refresh_token": refresh(now, 15 * 86400), "sub": "user@wayne",
               "issuer": "https://iss.example.org:8455", "scope": "storage.read:/"}
    w._save()
    assert status() == 0
    out = capsys.readouterr().out
    assert "user@wayne" in out and "storage.read:/" in out and "10 min left" in out
    assert "15.0 days" in out
    w.state.update(expires_at=0, refresh_token=refresh(now - 20 * 86400, 15 * 86400))
    w._save()
    assert status() == 1 and "./js_osdf.py login" in capsys.readouterr().out
    monkeypatch.setenv("BEARER_TOKEN", _jwt(sub="robot", scope="storage.create:/",
                                            exp=int(now) + 7200))
    assert status() == 0 and "a bearer token comes first" in capsys.readouterr().out


def test_shared_login_renewed_once(tmp_path, monkeypatch):
    """Jobs sharing a login (one home directory): one renews, the others read its token,
    although the issuer accepts each refresh token once only."""
    import fcntl
    import multiprocessing
    monkeypatch.setattr(js_osdf, "WEB_DIR", str(tmp_path / "web"))
    issuer_state, calls = tmp_path / "issuer.json", tmp_path / "calls"
    issuer_state.write_text(json.dumps({"rt": "rt-0", "n": 0}))
    iss = "https://iss.example.org"

    def http(url, form=None, json_body=None, redirect=True, timeout=30, **kw):
        assert url == f"{iss}/token" and form["grant_type"] == "refresh_token"
        with open(issuer_state, "r+") as f:            # the issuer, across processes
            fcntl.flock(f, fcntl.LOCK_EX)
            st = json.load(f)
            with open(calls, "a") as c:
                c.write(form["refresh_token"] + "\n")
            if form["refresh_token"] != st["rt"]:
                return 400, {}, b'{"error": "invalid_grant"}'
            js_osdf.time.sleep(0.2)                    # the others are waiting meanwhile
            st["n"] += 1
            st["rt"] = f"rt-{st['n']}"
            f.seek(0), f.truncate(), json.dump(st, f), f.flush()
        return 200, {}, json.dumps({"access_token": _jwt(exp=int(js_osdf.time.time()) + 1200,
                                                         jti=st["n"]),
                                    "refresh_token": st["rt"]}).encode()
    monkeypatch.setattr(js_osdf, "_http", http)
    w = js_osdf.WebLogin("/fno4hic")
    w.state = {"access_token": "old", "expires_at": 0, "refresh_token": "rt-0",
               "token_endpoint": f"{iss}/token", "issuer": iss, "client_id": "c",
               "client_secret": "s", "scope_path": "/"}
    w._save()

    ctx = multiprocessing.get_context("fork")
    barrier, out = ctx.Barrier(6), ctx.Queue()

    def job():
        login = js_osdf.WebLogin("/fno4hic")           # read at the start, all expired
        barrier.wait()
        out.put(login.token() or f"FAILED {login.error}")
    procs = [ctx.Process(target=job) for _ in range(6)]
    for p in procs:
        p.start()
    tokens = [out.get(timeout=30) for _ in procs]
    for p in procs:
        p.join(timeout=30)
    assert calls.read_text().split() == ["rt-0"]       # renewed once, by one of them
    assert len(set(tokens)) == 1 and not tokens[0].startswith("FAILED")
    assert json.loads(w.file.read_text())["refresh_token"] == "rt-1"
    assert not list(w.file.parent.glob("*.tmp"))
    # a refused renewal says why
    w = js_osdf.WebLogin("/fno4hic")
    w.state.update(expires_at=0, refresh_token="rt-0")
    w._save()
    assert w.token() is None and "invalid_grant" in w.error


def test_refresh_falls_back_to_the_logins(tmp_path, monkeypatch):
    """The Wayne issuer (OA4MP) fails on refresh tokens from a renewal (HTTP 500, "Null
    pointer") but takes the login's again: renewals fall back to it."""
    monkeypatch.setattr(js_osdf, "WEB_DIR", str(tmp_path / "web"))
    sent, n = [], [0]

    def http(url, form=None, json_body=None, redirect=True, timeout=30, **kw):
        sent.append(form["refresh_token"])
        if form["refresh_token"] != "rt-login":
            return 500, {}, b'error="server_error"\nerror_description="Null+pointer"\n'
        n[0] += 1
        return 200, {}, json.dumps({"access_token": _jwt(exp=int(js_osdf.time.time()) + 1200,
                                                         jti=n[0]),
                                    "refresh_token": f"rt-renewed-{n[0]}"}).encode()
    monkeypatch.setattr(js_osdf, "_http", http)
    w = js_osdf.WebLogin("/fno4hic")
    w.state = {"access_token": "old", "expires_at": 0, "refresh_token": "rt-login",
               "token_endpoint": "https://iss/token", "client_id": "c", "client_secret": "s"}
    w._save()
    for i in range(3):
        w.state["expires_at"] = 0
        w._save()
        assert w.token() is not None, w.error
    assert sent == ["rt-login", "rt-renewed-1", "rt-login", "rt-renewed-2", "rt-login"]
    saved = json.loads(w.file.read_text())
    assert saved["login_refresh_token"] == "rt-login" and saved["refresh_token"] == "rt-renewed-3"
    # both refused: the error names both answers
    w.state.update(expires_at=0, login_refresh_token="rt-gone")
    w._save()
    assert w.token() is None and w.error.count("Null pointer") == 2
