"""
_common/tutorial_env.py -- what every PyJetscape tutorial does first.

    import sys; sys.path.insert(0, ".")        # the tutorial folder
    from _common import tutorial_env as te
    env = te.setup()                           # detects the setup, registers HDF5 filters
    path = te.data("oo_test", "OO_workflow_hadrons_bulk_bg.h5")   # local file, else HTTPS URL
    with te.h5("oo_test", "OO_workflow_hadrons_bulk_bg.h5") as f: ...

It travels with the tutorials (the tarball, the checkout, later the container and the wheel),
not with the jetscape package, so the tutorials run with the releases already out: the
production image from 2026-10-28 on, the xscape wheel 0.2.0, the analysis env, Colab.

The setups (platform()):
    container   the production image ($XSCAPE=/opt/X-SCAPE): runs X-SCAPE, no ROOT
    mac-wheel   the xscape wheel (macOS): runs X-SCAPE, no ROOT
    source      a checkout with the compiled core (a source build)
    analysis    the jetscape readers without the core (analysis env, pyjetscape wheel)
    colab       Google Colab: analysis only, set up by colab_install()
    bare        no jetscape package yet

Data (data(), h5()): a file of a set in datasets.json is taken from the local data
directory ($JS_TUTORIAL_DATA, else data/ in the tutorial folder) when it is there, else
streamed over HTTPS from the OSDF (no login).  get_data.py downloads sets.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform as _platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]          # the tutorial folder (pyjetscape/)
DATASETS = HERE / "datasets.json"
TUTORIALS = HERE / "tutorials.json"

#: what Colab installs: the pure-Python jetscape package with the analysis packages
PYJETSCAPE_WHEEL = ("pyjetscape[analysis] @ https://github.com/jhputschke/xscape-macos/"
                    "releases/download/v0.2.0/pyjetscape-1.4-py3-none-any.whl")
COLAB_EXTRA = ["hdf5plugin", "fsspec", "aiohttp", "requests"]

#: the tutorials as a tarball: a release with its own tag (tutorials-vN, never "latest",
#: which belongs to js-contrib's code releases), and the main branch (what Colab opens)
TARBALL_URL = ("https://github.com/jhputschke/js-contrib/releases/download/{tag}/"
               "pyjetscape-tutorials.tar.gz")
BRANCH_ARCHIVE = "https://codeload.github.com/jhputschke/js-contrib/tar.gz/refs/heads/main"

#: what a tutorial's "needs" (tutorials.json) mean: capability names that must all be there
REQUIREMENTS = {
    "analysis": ["jetscape", "h5py", "hdf5plugin", "numpy", "matplotlib", "hist", "mplhep"],
    "fastjet": ["fastjet", "awkward"],
    "xscape": ["core"],
    "root": ["pyroot"],
    "torch": ["torch"],
    "viz": ["pyvista"],
}
NEEDS_TEXT = {
    "analysis": "the jetscape readers",
    "fastjet": "FastJet (Python)",
    "xscape": "X-SCAPE (container, Mac wheel or source build)",
    "root": "ROOT with PyROOT (analysis env + conda ROOT)",
    "torch": "PyTorch (source build)",
    "viz": "PyVista (analysis env, or the -viz image)",
}

_env = None


# ── the setup ───────────────────────────────────────────────────────────────────────────
def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def in_colab() -> bool:
    return "google.colab" in sys.modules


def platform() -> str:
    """container, mac-wheel, source, analysis, colab or bare (see the module docstring)."""
    if in_colab():
        return "colab"
    if not _has("jetscape"):
        return "bare"
    if os.environ.get("XSCAPE") == "/opt/X-SCAPE" and Path("/opt/X-SCAPE").is_dir():
        return "container"
    try:
        from jetscape import paths
        if paths.installed_tree() is not None:
            return "mac-wheel"
    except Exception:
        pass
    import jetscape
    return "source" if getattr(jetscape, "HAS_CORE", False) else "analysis"


def capabilities(check_root: bool = True) -> dict:
    """{name: (available, detail)} for the packages and services the tutorials use."""
    caps = {}
    try:
        import jetscape
        core = bool(getattr(jetscape, "HAS_CORE", False))
        loc = Path(jetscape.__file__).parent
        caps["jetscape"] = (True, str(loc))
        caps["core"] = (core, "runs X-SCAPE" if core else "readers only (no compiled core)")
    except Exception as e:                           # noqa: BLE001
        caps["jetscape"] = (False, f"{type(e).__name__}: {e}")
        caps["core"] = (False, "no jetscape package")
    caps["gpu_hydro"] = hydro_gpu()
    for name in ("numpy", "h5py", "hdf5plugin", "matplotlib", "hist", "mplhep", "awkward",
                 "fastjet", "uproot", "fsspec", "aiohttp", "pelicanfs", "torch", "pyvista"):
        ok = _has(name)
        ver = ""
        if ok:
            try:
                from importlib.metadata import version
                ver = version({"fastjet": "fastjet", "pyvista": "pyvista"}.get(name, name))
            except Exception:                        # noqa: BLE001
                ver = "installed"
        caps[name] = (ok, ver or "not installed")
    if check_root:
        ok, detail = False, "not installed"
        if _has("ROOT"):
            try:
                ROOT = importlib.import_module("ROOT")
                ok, detail = True, ROOT.gROOT.GetVersion()
            except Exception as e:                   # noqa: BLE001
                detail = f"import failed: {e}"
        caps["pyroot"] = (ok, detail)
    return caps


def hydro_gpu() -> tuple[bool, str]:
    """Where MUSIC (the hydro) runs -> (on a GPU, detail), from the compiled core's
    MUSIC_BACKEND: "music4gpu-metal" uses the Apple Silicon GPU by default (the Mac wheel),
    "music4gpu-cuda" an NVIDIA GPU when the container sees one (--gpus all / --nv), else
    MUSIC's CPU path; "music" is the CPU MUSIC."""
    try:
        from jetscape.pyjetscape_core import MUSIC_BACKEND as backend
    except Exception:                                # noqa: BLE001
        return False, "no compiled core: X-SCAPE does not run here"
    if backend == "music4gpu-metal":
        if sys.platform == "darwin" and _platform.machine() == "arm64":
            return True, "Metal: the Apple Silicon GPU, used by default"
        return False, "Metal build, but no Apple Silicon GPU here: MUSIC runs on the CPU"
    if backend == "music4gpu-cuda":
        try:
            r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
            gpus = [g for g in r.stdout.splitlines() if g.startswith("GPU")]
        except (OSError, subprocess.SubprocessError):
            gpus = []
        if gpus:
            return True, f"CUDA: {gpus[0].split(' (UUID')[0]}" + \
                (f" (+{len(gpus) - 1} more)" if len(gpus) > 1 else "")
        return False, ("CUDA build, but no NVIDIA GPU visible (Docker: --gpus all, "
                       "Apptainer: --nv): MUSIC runs on the CPU, about 10x slower")
    return False, f"{backend}: MUSIC runs on the CPU"


def meets(needs, caps: dict) -> tuple[bool, list[str]]:
    """Whether ``needs`` (tutorials.json) are met -> (ok, the missing capabilities)."""
    missing = [c for n in needs for c in REQUIREMENTS.get(n, [n])
               if not caps.get(c, (False,))[0]]
    return not missing, missing


class Env:
    def __init__(self, platform: str, caps: dict, data_dir: Path):
        self.platform, self.caps, self.data_dir = platform, caps, data_dir

    def has(self, need: str) -> bool:
        return meets([need], self.caps)[0]

    def __repr__(self):
        return f"Env(platform={self.platform!r}, data_dir={str(self.data_dir)!r})"


def setup(quiet: bool = False, check_root: bool = False) -> Env:
    """Detect the setup, register the HDF5 compression filters (the files use Blosc: without
    hdf5plugin h5py reads the metadata but no data), and print one line about it."""
    global _env
    try:
        import hdf5plugin  # noqa: F401  (registers Blosc and friends with h5py)
    except ImportError:
        pass
    _env = Env(platform(), capabilities(check_root=check_root), data_dir())
    if not quiet:
        core = ("runs X-SCAPE; hydro: " + _env.caps["gpu_hydro"][1].split(":")[0]
                + (" GPU" if _env.caps["gpu_hydro"][0] else "")
                if _env.caps["core"][0] else
                "analysis only" if _env.caps["jetscape"][0] else "no jetscape package")
        print(f"tutorials {version()} | setup: {_env.platform} ({core}) | Python "
              f"{_platform.python_version()} | data: {_env.data_dir}")
        if not _env.caps["hdf5plugin"][0]:
            print("  warning: hdf5plugin is missing -- h5py cannot read the compressed "
                  "datasets (pip install hdf5plugin)")
    return _env


def version() -> str:
    try:
        return (HERE / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def colab_install(quiet: bool = True) -> None:
    """In Colab: install the jetscape readers and the analysis packages (about a minute)."""
    if not in_colab() or _has("jetscape"):
        return
    cmd = [sys.executable, "-m", "pip", "install", "-q", PYJETSCAPE_WHEEL, *COLAB_EXTRA]
    print("Installing the jetscape readers and the analysis packages ...")
    subprocess.run(cmd, check=True, capture_output=quiet)
    importlib.invalidate_caches()


# ── the data ────────────────────────────────────────────────────────────────────────────
def datasets() -> dict:
    with open(DATASETS) as f:
        return json.load(f)


def data_dir() -> Path:
    env = os.environ.get("JS_TUTORIAL_DATA")
    return Path(env).expanduser() if env else HERE / "data"


def url(set_name: str, filename: str = "") -> str:
    """The HTTPS URL of a file of a set on the OSDF."""
    cfg = datasets()
    base = cfg["osdf_https"].rstrip("/") + "/" + cfg["namespace"].strip("/")
    return f"{base}/{cfg['sets'][set_name]['path'].strip('/')}/{filename}"


def local(set_name: str, filename: str) -> Path | None:
    p = data_dir() / set_name / filename
    return p if p.is_file() else None


def data(set_name: str, filename: str, stream: bool | None = None) -> str:
    """The file to read: the local copy when there is one (and ``stream`` is not True), else
    its HTTPS URL.  uproot opens either; for h5py use h5()."""
    p = None if stream else local(set_name, filename)
    if p is not None:
        return str(p)
    if stream is False:
        raise FileNotFoundError(f"{set_name}/{filename} not in {data_dir()}; "
                                f"python get_data.py {set_name}")
    return url(set_name, filename)


def h5(set_name: str, filename: str, stream: bool | None = None, block_mb: int = 4):
    """h5py.File of a set's file: local, or streamed over HTTPS in ``block_mb`` blocks
    (fsspec; only the blocks read are fetched)."""
    import h5py
    src = data(set_name, filename, stream)
    if src.startswith("https://"):
        import fsspec
        f = fsspec.filesystem("https").open(src, "rb", block_size=block_mb << 20)
        return h5py.File(f, "r")
    return h5py.File(src, "r")


def check_osdf(timeout: float = 30) -> dict:
    """Fetch the namespace's test object over HTTPS -> {ok, cache, seconds, error}."""
    cfg = datasets()
    u = cfg["osdf_https"].rstrip("/") + "/" + cfg["namespace"].strip("/") + \
        "/test/test-hello-world.txt"
    t0 = time.time()
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "js-contrib-tutorials"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            return {"ok": True, "cache": r.url.split("/")[2], "seconds": time.time() - t0,
                    "error": ""}
    except Exception as e:                           # noqa: BLE001
        return {"ok": False, "cache": "", "seconds": time.time() - t0, "error": str(e)}


# ── reports ─────────────────────────────────────────────────────────────────────────────
def _table(rows, header):
    """A Markdown table in Jupyter, plain text elsewhere."""
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, display
        if get_ipython() is not None:
            md = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
            md += "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)
            display(Markdown(md))
            return
    except ImportError:
        pass
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for r in [header, *rows]:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def report_capabilities(env: Env | None = None) -> None:
    env = env or _env or setup(quiet=True)
    rows = [(name, "yes" if ok else "no", detail) for name, (ok, detail) in env.caps.items()]
    _table(rows, ("component", "available", "version / detail"))


def report_datasets() -> None:
    rows = []
    for name, s in datasets()["sets"].items():
        where = "local" if (data_dir() / name).is_dir() and any((data_dir() / name).iterdir()) \
            else ("OSDF" if s["status"] == "available" else "planned")
        rows.append((name, s["size"], where, s["description"]))
    _table(rows, ("set", "size", "where", "contents"))


def report_tutorials(env: Env | None = None) -> None:
    env = env or _env or setup(quiet=True)
    with open(TUTORIALS) as f:
        tuts = json.load(f)["tutorials"]
    rows = []
    for t in tuts:
        ok, missing = meets(t["needs"], env.caps)
        if t["status"] != "ready":
            state = "planned"
        else:
            state = "runs here" if ok else "missing: " + ", ".join(missing)
        needs = "; ".join(NEEDS_TEXT.get(n, n) for n in t["needs"]) or "-"
        rows.append((t["id"], t["title"], needs, "yes" if ok else "no", state))
    _table(rows, ("#", "tutorial", "needs", "possible here", "status"))
