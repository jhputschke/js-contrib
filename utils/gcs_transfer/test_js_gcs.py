"""
utils/gcs_transfer/test_js_gcs.py

js_gcs.py without the network: file kinds, --what, the key lookup, and upload / download
against an in-memory bucket (skip what is there, incomplete HDF5, .part, CRC32C).

    pytest utils/gcs_transfer/test_js_gcs.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip("google_crc32c")
h5py = pytest.importorskip("h5py")

import js_gcs  # noqa: E402

STEM = "AuAu_jet_0001"
PROD = {f"{STEM}.h5": "pair", f"{STEM}.json": "pair", f"{STEM}.xml": "pair",
        f"{STEM}.log": "pair", f"{STEM}_particlize.h5": "h5",
        f"{STEM}_hadrons_bulk_jet.h5": "h5", f"{STEM}_hadrons_bulk_bg.h5": "h5",
        f"{STEM}_hadrons_jet_frag.h5": "h5", f"{STEM}_hadronize.log": "h5",
        f"{STEM}_hadrons.root": "root", "pth10-40_campaign.root": "root",
        "wake_observables.h5": "other", "run_jobs.campaign": "other"}


class FakeGcs:
    """js_gcs.Gcs on a dict: name -> bytes."""

    def __init__(self):
        self.bucket_name, self.objects, self.uploads, self.downloads = "test_fno", {}, [], []

    def list(self, prefix, directory=True):
        p = prefix.strip("/") + "/" if directory else prefix
        return {n: (len(b), self._crc(b)) for n, b in self.objects.items() if n.startswith(p)}

    def stat(self, name):
        b = self.objects.get(name)
        return None if b is None else (len(b), self._crc(b))

    @staticmethod
    def _crc(b):
        import base64

        import google_crc32c
        return base64.b64encode(google_crc32c.Checksum(b).digest()).decode()

    def upload(self, path, name):
        self.uploads.append(name)
        self.objects[name] = Path(path).read_bytes()

    def download(self, name, path):
        self.downloads.append(name)
        tmp = Path(f"{path}.part")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(self.objects[name])
        tmp.replace(path)


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


def _args(argv):
    return js_gcs.parse_args(argv)


def test_kinds_and_what():
    assert js_gcs.classify(list(PROD) + ["a.part", ".hidden"]) == PROD
    # remote listings: per directory
    assert js_gcs.classify([f"x/{n}" for n in PROD]) == {f"x/{n}": k for n, k in PROD.items()}
    assert js_gcs.classify(["x/A.h5", "y/A.json"]) == {"x/A.h5": "other", "y/A.json": "other"}
    assert js_gcs.parse_what("pair,root") == {"pair", "root"}
    assert js_gcs.parse_what("all") == set(js_gcs.KINDS)
    with pytest.raises(Exception, match="not pair, h5, root or all"):
        js_gcs.parse_what("hdf5")
    assert js_gcs.bucket_name("gs://test_fno/") == "test_fno"


def test_crc32c_as_gcs(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"hello world")
    assert js_gcs.crc32c_b64(p) == "yZRlqg=="       # GCS's crc32c of "hello world"
    assert js_gcs.same(p, (11, "yZRlqg=="), False)
    assert not js_gcs.same(p, (11, "AAAAAA=="), False)
    assert js_gcs.same(p, (11, "AAAAAA=="), True)   # --size-only
    assert not js_gcs.same(p, (12, "yZRlqg=="), True)


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


@pytest.mark.parametrize("what,kinds", [("pair", {"pair"}), ("h5", {"h5"}),
                                        ("root", {"root"}), ("pair,root", {"pair", "root"}),
                                        ("all", {"pair", "h5", "root", "other"})])
def test_upload_what(tmp_path, what, kinds, capsys):
    d = _prod(tmp_path)
    gcs = FakeGcs()
    assert js_gcs.cmd_upload(_args(["upload", str(d), "--what", what, "--prefix", "p"]),
                             gcs) == 0
    assert sorted(gcs.objects) == sorted(f"p/AuAu_a/{n}" for n, k in PROD.items()
                                         if k in kinds)
    for name, data in gcs.objects.items():
        assert data == (d / name.rsplit("/", 1)[1]).read_bytes()


def test_upload_skip_incomplete_force(tmp_path, capsys):
    d = _prod(tmp_path, incomplete=(f"{STEM}_hadrons_jet_frag.h5",))
    gcs = FakeGcs()
    assert js_gcs.cmd_upload(_args(["upload", str(d), "--what", "h5", "--dry-run"]), gcs) == 0
    assert gcs.uploads == [] and "would upload" in capsys.readouterr().out
    assert js_gcs.cmd_upload(_args(["upload", str(d), "--what", "h5"]), gcs) == 0
    out = capsys.readouterr().out
    assert "1 incomplete HDF5 file(s) left out" in out and "4 to upload" in out
    assert f"AuAu_a/{STEM}_hadrons_jet_frag.h5" not in gcs.objects
    # again: everything there is skipped; a changed file (same size) goes up again
    gcs.uploads.clear()
    p = d / f"{STEM}_hadronize.log"
    p.write_text(p.read_text().upper())
    assert js_gcs.cmd_upload(_args(["upload", str(d), "--what", "h5"]), gcs) == 0
    assert gcs.uploads == [f"AuAu_a/{STEM}_hadronize.log"]
    assert "3 already there" in capsys.readouterr().out
    gcs.uploads.clear()
    p.write_text(p.read_text().lower())                # --size-only: not seen
    js_gcs.cmd_upload(_args(["upload", str(d), "--what", "h5", "--size-only"]), gcs)
    assert gcs.uploads == []
    js_gcs.cmd_upload(_args(["upload", str(d), "--what", "h5", "--force",
                             "--include-incomplete"]), gcs)
    assert len(gcs.uploads) == 5


def test_download(tmp_path, capsys):
    d = _prod(tmp_path)
    gcs = FakeGcs()
    js_gcs.cmd_upload(_args(["upload", str(d), "--prefix", "prod"]), gcs)
    to = tmp_path / "local"
    assert js_gcs.cmd_download(_args(["download", "gs://test_fno/prod/AuAu_a", "--to", str(to),
                                      "--what", "root"]), gcs) == 0
    got = sorted(p.name for p in (to / "AuAu_a").iterdir())
    assert got == sorted(n for n, k in PROD.items() if k == "root")
    assert not list(to.rglob("*.part"))
    gcs.downloads.clear()
    assert js_gcs.cmd_download(_args(["download", "prod/AuAu_a", "--to", str(to)]), gcs) == 0
    assert len(gcs.downloads) == len(PROD) - 2          # the root files are there already
    for n in PROD:
        assert (to / "AuAu_a" / n).read_bytes() == (d / n).read_bytes()
    with pytest.raises(SystemExit, match="not in bucket"):
        js_gcs.cmd_download(_args(["download", "gs://other/prod/AuAu_a"]), gcs)
    with pytest.raises(SystemExit, match="nothing at"):
        js_gcs.cmd_download(_args(["download", "prod/missing"]), gcs)


def test_upload_files(tmp_path, capsys):
    d = _prod(tmp_path, incomplete=(f"{STEM}_hadrons_jet_frag.h5",))
    gcs = FakeGcs()
    # named files, whatever --what says; next to where their directory goes, or --flat
    assert js_gcs.cmd_upload(_args(["upload", str(d / f"{STEM}.json"),
                                    str(d / "run_jobs.campaign"), "--what", "root",
                                    "--prefix", "p"]), gcs) == 0
    assert sorted(gcs.objects) == [f"p/AuAu_a/{STEM}.json", "p/AuAu_a/run_jobs.campaign"]
    assert f"({'pair'}) -> gs://test_fno/p/AuAu_a/{STEM}.json" in capsys.readouterr().out
    js_gcs.cmd_upload(_args(["upload", str(d / f"{STEM}.xml"), "--flat"]), gcs)
    js_gcs.cmd_upload(_args(["upload", str(d / f"{STEM}.xml"), "--flat", "--prefix", "q"]),
                      gcs)
    assert {f"{STEM}.xml", f"q/{STEM}.xml"} <= set(gcs.objects)
    # skipped when there, incomplete and .part left out, a directory and a file in it once
    gcs.uploads.clear()
    js_gcs.cmd_upload(_args(["upload", str(d / f"{STEM}.json"),
                             str(d / f"{STEM}_hadrons_jet_frag.h5"),
                             str(d / f"{STEM}_hadrons.root.part"), "--prefix", "p"]), gcs)
    out = capsys.readouterr().out
    assert gcs.uploads == [] and "1 already there" in out and "1 incomplete" in out
    assert "left out (temporary or hidden file)" in out
    js_gcs.cmd_upload(_args(["upload", str(d), str(d / f"{STEM}.log"), "--what", "pair"]),
                      gcs)
    assert sorted(gcs.uploads) == sorted(f"AuAu_a/{n}" for n, k in PROD.items()
                                         if k == "pair")
    with pytest.raises(SystemExit, match="not found"):
        js_gcs.cmd_upload(_args(["upload", str(d / "missing.h5")]), gcs)


def test_download_files_and_patterns(tmp_path, capsys):
    d = _prod(tmp_path)
    gcs = FakeGcs()
    js_gcs.cmd_upload(_args(["upload", str(d), "--prefix", "prod"]), gcs)
    to = tmp_path / "local"
    # one file, and a pattern (with --what) -> TO/<file>
    assert js_gcs.cmd_download(_args(["download", f"gs://test_fno/prod/AuAu_a/{STEM}.xml",
                                      f"prod/AuAu_a/{STEM}_hadrons*", "--what", "h5",
                                      "--to", str(to)]), gcs) == 0
    assert sorted(p.name for p in to.iterdir()) == sorted(
        [f"{STEM}.xml"] + [f"{STEM}_hadrons_{t}.h5" for t in js_gcs.HADRON_TAGS])
    assert (to / f"{STEM}.xml").read_bytes() == (d / f"{STEM}.xml").read_bytes()
    assert "3 of 4 matching file(s) selected" in capsys.readouterr().out   # not the .root
    gcs.downloads.clear()
    js_gcs.cmd_download(_args(["download", f"prod/AuAu_a/{STEM}.x?l", "--to", str(to)]), gcs)
    assert gcs.downloads == [] and "1 already here" in capsys.readouterr().out
    # two objects to one local path: refused
    gcs.objects[f"other/{STEM}.xml"] = b"x"
    with pytest.raises(SystemExit, match="would both be"):
        js_gcs.cmd_download(_args(["download", f"*/{STEM}.xml", "--to", str(to)]), gcs)
    with pytest.raises(SystemExit, match="not the whole bucket"):
        js_gcs.cmd_download(_args(["download", "gs://test_fno/"]), gcs)


def test_remove_env(tmp_path, monkeypatch, capsys):
    env = tmp_path / "js_gcs" / "venv"
    monkeypatch.setattr(js_gcs, "ENV_DIR", env)
    assert js_gcs.remove_env() == 0 and "no environment" in capsys.readouterr().out
    env.mkdir(parents=True)
    (env / "keep.txt").write_text("not a venv")
    assert js_gcs.remove_env() == 1 and (env / "keep.txt").exists()   # left alone
    (env / "pyvenv.cfg").write_text("")
    (env / js_gcs.STAMP).write_text("")
    assert js_gcs.remove_env() == 0
    assert not env.exists() and not env.parent.exists()
