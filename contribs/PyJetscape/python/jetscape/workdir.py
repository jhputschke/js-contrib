"""
python/jetscape/workdir.py

A job's private working directory and its seed, for the production scripts and for scripts
and notebooks of one's own (docs/Plans/PLAN_container_user.md, step 1).  Pure Python, no
build needed to import.

The functions moved here unchanged from example/prod_AuAu_0_10/run_prod.py, which re-exports
them under the same names, so every production script keeps calling ``rp.enter_workdir(a, ...)``
etc.  They take the scripts' parsed arguments ``a`` (any object with the attributes they
read: ``build``, ``main_xml``, ``workdir``, ``in_build``, ``keep_workdir``, ``seed``, ...).

For a script of one's own, JobDir does the same with plain arguments:

    import jetscape as js
    from jetscape.workdir import JobDir

    with JobDir("out", "my_job.xml") as job:            # -> out/my_job.xml, out/work/my_job
        with js.per_event(job.main_xml, job.user_xml) as driver:
            for i in driver.events():
                ...
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import socket
import sys
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace

from . import paths

__all__ = ["JobDir", "WORKDIR_LINKS", "enter_workdir", "job_workdir", "leave_workdir",
           "add_workdir_args", "SEED_MAX", "add_seed_args", "seed_args_problems",
           "output_name", "seed_registry", "resolve_seed", "seed_provenance",
           "pythia_data_problem", "hydro_label", "build_or_job_dir"]


# ─────────────────────────────────────────────────────────── working directory
#: read-only assets still opened relative to the working directory: the vendored
#: 3dMCGlauber / trento code, and Matter, whose read_tables() hard-codes
#: "LBT-tables/{ratedata-HQ,distB.dat,distF.dat}" (LBT itself honours LBT_TABLES_PATH).
#: Without LBT-tables, Matter's heavy-quark recoil sampling (collHQ22) runs on all-zero
#: tables.  X-SCAPE's examples/run_in_workdir.sh links the same set minus LBT-tables.
WORKDIR_LINKS = ("tables", "eps09", "LHAPDF_Lib", "nucleusConfigs", "data_table",
                 "LBT-tables")


# ---------------------------------------------------------------------------------- seeds
#: Pythia takes seeds 1..SEED_MAX and silently clamps larger ones to SEED_MAX, so jobs with
#: larger seeds (a date, Unix time) would all get the same Pythia stream: the same jets.
SEED_MAX = 900_000_000
_CAMPAIGN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REGISTRY_HEADER = "# seed\tsource\tcampaign\tfile\thost\tdate\n"


def add_seed_args(p: argparse.ArgumentParser, drives: str) -> None:
    """--seed, --campaign, --index and --seed-registry (run_prod.py and run_prod_jet.py)."""
    p.add_argument("--seed", type=int, default=1,
                   help=f"framework seed (drives {drives}), 1..{SEED_MAX}; the same seed "
                        "gives the same collisions. 0: draw a unique one from OS entropy, "
                        "recorded in the file, the .json and the seed registry (the job stays "
                        "reproducible); the file is then named by --campaign. Not X-SCAPE's "
                        "own seed 0, which Pythia takes from the clock in seconds (jobs "
                        "started together get the same jets) and records nowhere")
    p.add_argument("--campaign", default=None,
                   help="put the campaign in the name: <base>_<CAMPAIGN>_<NNNN>.h5 with --seed 0 "
                        "(NNNN = --index), <base>_<CAMPAIGN>_seed<NNNN>.h5 with an explicit "
                        "seed (NNNN = the seed); letters, digits, '.', '_', '-'. Default with "
                        "--seed 0: the start time, YYYYMMDD-HHMMSS")
    p.add_argument("--index", type=int, default=1,
                   help="job number within the campaign: NNNN in the name with --seed 0, "
                        "recorded as prod_index (default 1)")
    p.add_argument("--seed-registry", default=None, dest="seed_registry",
                   help="file of the seeds already used (tab-separated, appended per job); "
                        "--seed 0 draws only seeds not in it. Default: seeds_used.tsv next "
                        "to OUTDIR, shared by the campaigns kept side by side there; 'none' "
                        "turns it off")


def seed_args_problems(a) -> list:
    problems = []
    if not 0 <= a.seed <= SEED_MAX:
        problems.append(f"--seed must be 0 (unique, from OS entropy) or 1..{SEED_MAX}, got "
                        f"{a.seed}: Pythia (the jet production's hard process) clamps "
                        f"larger seeds to {SEED_MAX}, so every such job would get the same "
                        "jets; the same range holds everywhere")
    if a.index < 1:
        problems.append(f"--index must be >= 1, got {a.index}")
    if a.campaign is not None and not _CAMPAIGN_RE.match(a.campaign):
        problems.append(f"--campaign {a.campaign!r}: use letters, digits, '.', '_', '-' "
                        "(no '/', not starting with '.' or '-')")
    return problems


def output_name(a, base: str) -> str:
    """The output file name: --out, else by seed when the seed is given --
    <base>[_<campaign>]_seed<NNNN>.h5 -- and by job number when it is drawn (--seed 0, with
    the start time as the campaign if none was given): <base>_<campaign>_<NNNN>.h5."""
    if a.campaign is None and a.seed == 0:
        a.campaign = time.strftime("%Y%m%d-%H%M%S")
    if a.out:
        return a.out
    if a.seed == 0:
        return f"{base}_{a.campaign}_{a.index:04d}.h5"
    return f"{base}{'_' + a.campaign if a.campaign else ''}_seed{a.seed:04d}.h5"


def seed_registry(a):
    if a.seed_registry and a.seed_registry.lower() == "none":
        return None
    return os.path.abspath(a.seed_registry or
                           os.path.join(os.path.dirname(os.path.abspath(a.outdir)),
                                        "seeds_used.tsv"))


def resolve_seed(a, out_h5: str, record: bool = True) -> None:
    """Make ``a.seed`` concrete and note where it came from (``a.seed_source``).

    --seed 0 draws from OS entropy (``secrets``) a seed in 1..SEED_MAX that the registry does
    not hold yet.  With ``record`` the seed is appended to the registry (under an exclusive
    lock, so jobs starting together cannot draw the same one); an explicit seed is recorded
    too, and reported when the registry has it for another file (the same collisions: fine for
    paired comparisons, not for more statistics)."""
    import fcntl
    import secrets

    path = seed_registry(a)
    a.seed_source = "os_entropy" if a.seed == 0 else "explicit"
    if path is None or (not record and not os.path.exists(path)):
        if a.seed == 0:
            a.seed = secrets.randbelow(SEED_MAX) + 1
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        used = {}
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if line.startswith("#") or len(cols) < 4 or not cols[0].isdigit():
                continue
            used.setdefault(int(cols[0]), []).append(cols[3])
        if a.seed == 0:
            while True:
                seed = secrets.randbelow(SEED_MAX) + 1
                if seed not in used:
                    break
            a.seed = seed
        else:
            others = [f for f in used.get(a.seed, []) if f != os.path.abspath(out_h5)]
            if others:
                print(f"note: seed {a.seed} was used before, for {others[0]}"
                      f"{f' (+{len(others) - 1} more)' if len(others) > 1 else ''}: the same "
                      "collisions -- fine for comparing settings, not for more statistics",
                      file=sys.stderr)
        if record and os.path.abspath(out_h5) not in used.get(a.seed, []):
            if fh.tell() == 0:
                fh.write(_REGISTRY_HEADER)
            fh.write("\t".join([str(a.seed), a.seed_source, a.campaign or "",
                                 os.path.abspath(out_h5), socket.gethostname(),
                                 time.strftime("%Y-%m-%dT%H:%M:%S")]) + "\n")
            fh.flush()


def seed_provenance(a) -> dict:
    return {"prod_seed": a.seed, "prod_seed_source": a.seed_source,
            "prod_campaign": a.campaign or "", "prod_index": a.index}


def add_workdir_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workdir", default=None,
                   help="this job's working directory (default: OUTDIR/work/<output name>)")
    p.add_argument("--keep-workdir", action="store_true", dest="keep_workdir",
                   help="keep the working directory after a successful job (it holds only "
                        "the modules' side files: 3dMCGlauber strings_event_*.dat, MUSIC's "
                        "momentum_anisotropy_*.dat, ...); a failed job always keeps it")
    p.add_argument("--in-build", action="store_true", dest="in_build",
                   help="run in the build tree itself, as before (shares music_input and "
                        "every side file with any other job running there)")


#: what an installed tree keeps directly in its data folder that jetscape_main.xml finds
#: through ../external_packages/iSS/ (see _absolute_parent_paths)
INSTALLED_ASSETS = ("iSS_tables", "iSS_parameters.dat")


def _absolute_parent_paths(src: str, dst: str, build: str) -> None:
    """Copy an XML, making every element text that starts with '../' absolute.

    Such paths (iSS, SMASH, hydro-from-file, Martini, ... in jetscape_main.xml) are
    relative to the build tree, which is the working directory only with --in-build.
    './' paths (outputs such as ./FinalPartonsInfo.dat, iSS_working_path '.') stay
    relative: they belong in the job's own directory.

    iSS's '../' paths with no target fall back to the build's iSS_tables and
    iSS_parameters.dat (INSTALLED_ASSETS) if those exist: an installed tree (the xscape
    wheel's share/xscape) has them but no ../external_packages/iSS -- X-SCAPE's
    run_in_workdir.sh points the iSS entries there too.  In a checkout or the image the
    '../' targets exist.  Only these names: another missing target (the test hydro files)
    stays as it was, and macOS would match MUSIC_input to the build's music_input.
    """
    tree = ET.parse(src)
    for el in tree.iter():
        text = (el.text or "").strip()
        if text.startswith("../"):
            path = os.path.normpath(os.path.join(build, text))
            name = os.path.basename(path)
            beside = os.path.join(build, name)
            if name in INSTALLED_ASSETS and not os.path.exists(path) and os.path.exists(beside):
                path = beside
            el.text = path
    tree.write(dst)


def enter_workdir(a, workdir: str, job_xml_path: str) -> str:
    """Make ``workdir`` this job's working directory -> the main XML to use.

    The Python counterpart of X-SCAPE's examples/run_in_workdir.sh (which execs the
    runJetscape binary, so it cannot drive PyJetscape).  In the build tree, concurrent jobs
    share music_input -- MusicWrapper truncates and rewrites it at every MUSIC init, and a
    job reading it in that window spins forever in MUSIC's StringFind4 -- as well as every
    file the modules write to the working directory (3dMCGlauber strings_event_<N>.dat and
    events_summary.dat, MUSIC's momentum_anisotropy / eccentricities / meanpT files, ...).
    Here each job has its own directory with a private music_input, and the read-only
    assets come from the build tree:

    * ``XSCAPE_DATA_DIR`` -> mcglauber.input (MCGlauberWrapper), LBT-tables fallback
    * ``HYDROPROGRAMPATH`` -> MUSIC's EOS tables
    * ``LBT_TABLES_PATH`` -> LBT tables
    * symlinks for the dirs still opened relative to the working directory (WORKDIR_LINKS)
    * '../' paths in the main and job XML made absolute (see _absolute_parent_paths)
    """
    build = a.build
    os.makedirs(workdir, exist_ok=True)
    shutil.copyfile(os.path.join(build, "music_input"), os.path.join(workdir, "music_input"))
    for name in WORKDIR_LINKS:
        src, dst = os.path.join(build, name), os.path.join(workdir, name)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    os.environ["XSCAPE_DATA_DIR"] = build
    os.environ["HYDROPROGRAMPATH"] = build
    os.environ["LBT_TABLES_PATH"] = os.path.join(build, "LBT-tables")
    main_xml = os.path.join(workdir, "jetscape_main.xml")
    _absolute_parent_paths(a.main_xml, main_xml, build)
    _absolute_parent_paths(job_xml_path, job_xml_path, build)
    os.chdir(workdir)
    return main_xml


def job_workdir(a, out_h5: str) -> str:
    """This job's working directory: --workdir, else OUTDIR/work/<output name>."""
    if a.workdir:
        return os.path.abspath(a.workdir)
    return os.path.join(os.path.dirname(out_h5), "work",
                        os.path.splitext(os.path.basename(out_h5))[0])


def leave_workdir(a, workdir: str, ok: bool) -> None:
    """Remove the working directory after a successful job unless --keep-workdir."""
    if a.in_build:
        return
    os.chdir(os.path.dirname(workdir))
    if ok and not a.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
        if not a.workdir:                # OUTDIR/work, once its last job is gone
            try:
                os.rmdir(os.path.dirname(workdir))
            except OSError:
                pass
    else:
        print(f"  working directory kept: {workdir}")


# ───────────────────────────────────────────────────────────────── job setup
def pythia_data_problem():
    """None if pyjetscape_core can be imported, else the reason.

    Pythia finds its xmldoc through PYTHIA8DATA or the path compiled into libpythia8.
    Homebrew's Pythia (macOS) has a working compiled-in path; a relocated conda Pythia
    (the js_fno env) may not, and then the import aborts the process -- so it is probed
    in a subprocess, and only when PYTHIA8DATA is unset.
    """
    if os.environ.get("PYTHIA8DATA"):
        return None
    import subprocess
    probe = (f"import sys; sys.path.insert(0, {str(paths.pyjetscape_dir() / 'python')!r}); "
             "import jetscape")
    try:
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                           timeout=300)
    except subprocess.TimeoutExpired:
        return "importing pyjetscape_core timed out (PYTHIA8DATA is not set)"
    if r.returncode == 0:
        return None
    tail = " | ".join((r.stderr or r.stdout).strip().splitlines()[-3:])
    return ("PYTHIA8DATA is not set and importing pyjetscape_core fails without it: point it "
            "to Pythia's xmldoc (`conda activate js_fno` sets it). "
            f"Import error (exit {r.returncode}): {tail}")


def hydro_label(build: str) -> str:
    """The production files' ``hydro`` attribute: "MUSIC (music4gpu)" or "MUSIC".

    From the compiled core's MUSIC_BACKEND (the MUSIC it was linked against: "music",
    "music4gpu-cuda", "music4gpu-metal"); a core without it (built before) falls back to
    the old test, "gpu" in the build folder's name (build_gpu).
    """
    try:
        from .pyjetscape_core import MUSIC_BACKEND
    except ImportError:
        MUSIC_BACKEND = None
    if MUSIC_BACKEND is None:
        return "MUSIC (music4gpu)" if "gpu" in os.path.basename(build) else "MUSIC"
    return "MUSIC (music4gpu)" if MUSIC_BACKEND.startswith("music4gpu") else "MUSIC"


# ──────────────────────────────────────────────────────── for one's own scripts
class JobDir:
    """A job's private working directory, as the production scripts make it, for a script
    or notebook of one's own -- runnable from any writable directory (a container's /work).

    On entering: ``user_xml`` is copied to ``<out_dir>/<name>.xml`` (``job.user_xml``), the
    main XML to ``<out_dir>/work/<name>/jetscape_main.xml`` (``job.main_xml``), their '../'
    paths made absolute against the build, music_input copied and the build's tables linked
    (enter_workdir), and the current directory changed to ``job.path``, where the modules
    write their side files (3dMCGlauber's events_summary.dat, ...).

    On leaving: back to the directory it was entered from; ``job.path`` is removed unless
    ``keep`` or an exception left the block (then it is kept and printed).

    ``build``: default jetscape.paths.build_dir() ($JETSCAPE_BUILD, else <X-SCAPE>/build_gpu);
    ``main_xml``: default jetscape.paths.main_xml(); ``name``: default the user XML's stem.
    """

    def __init__(self, out_dir, user_xml, *, name=None, main_xml=None, build=None,
                 keep=False):
        self.out_dir = os.path.abspath(out_dir)
        self.build = os.path.abspath(build or paths.build_dir())
        self.source_xml = os.path.abspath(user_xml)
        self.name = name or os.path.splitext(os.path.basename(self.source_xml))[0]
        self.source_main_xml = os.path.abspath(main_xml or paths.main_xml())
        self.keep = keep
        self.path = os.path.join(self.out_dir, "work", self.name)
        self.user_xml = os.path.join(self.out_dir, self.name + ".xml")
        self.main_xml = None
        self._cwd = None

    def __enter__(self) -> "JobDir":
        missing = [p for p in (os.path.join(self.build, "music_input"), self.source_xml,
                               self.source_main_xml) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "JobDir: not found: " + ", ".join(missing) + " (the build is $JETSCAPE_BUILD, "
                "else $XSCAPE/build_gpu; see jetscape.paths)")
        os.makedirs(self.out_dir, exist_ok=True)
        if self.source_xml != self.user_xml:
            shutil.copyfile(self.source_xml, self.user_xml)
        self._cwd = os.getcwd()
        a = SimpleNamespace(build=self.build, main_xml=self.source_main_xml)
        self.main_xml = enter_workdir(a, self.path, self.user_xml)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        os.chdir(self._cwd)
        if exc_type is None and not self.keep:
            shutil.rmtree(self.path, ignore_errors=True)
            try:
                os.rmdir(os.path.dirname(self.path))       # <out_dir>/work, if now empty
            except OSError:
                pass
        else:
            print(f"working directory kept: {self.path}")
        return False

    def __repr__(self) -> str:
        return f"JobDir({self.path!r}, build={self.build!r})"


@contextlib.contextmanager
def build_or_job_dir(out_dir, user_xml, main_xml=None, *, name=None, keep=False):
    """Where an example script runs X-SCAPE -> an object with ``main_xml``, ``user_xml``,
    ``path``.

    In a build tree (``music_input`` in the current directory: the old way, ``cd
    X-SCAPE/build_gpu`` and run the script from there) it runs right here with the XMLs as
    given, as these scripts always did.  Anywhere else -- a copy of the examples
    (xscape-examples), the container's /work, the macOS wheel -- it runs in a JobDir under
    ``out_dir`` (``keep``: leave its working directory, e.g. for files the XML writes there).
    Give output paths absolute: a JobDir changes the current directory.
    """
    main_xml = os.path.abspath(main_xml or paths.main_xml())
    if os.path.isfile("music_input"):
        yield SimpleNamespace(main_xml=main_xml, user_xml=os.path.abspath(user_xml),
                              path=os.getcwd())
        return
    with JobDir(out_dir, user_xml, name=name, main_xml=main_xml, keep=keep) as job:
        yield job
