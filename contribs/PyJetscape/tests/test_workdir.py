"""
tests/test_workdir.py

jetscape.paths and jetscape.workdir (PLAN_container_user.md, step 1): where the tree and the
build are, run_prod.py's re-exports, and JobDir on a stand-in build tree.  Nothing here needs
the compiled extension.

    pytest tests/test_workdir.py -q
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from jetscape import paths, workdir
from jetscape.workdir import JobDir

PYJETSCAPE = Path(os.path.abspath(__file__)).parents[1]


@pytest.fixture
def clean_env(monkeypatch):
    """No XSCAPE/JETSCAPE_BUILD from the caller; the variables enter_workdir sets are
    restored afterwards."""
    for k in ("XSCAPE", "JETSCAPE_BUILD"):
        monkeypatch.delenv(k, raising=False)
    for k in ("XSCAPE_DATA_DIR", "HYDROPROGRAMPATH", "LBT_TABLES_PATH"):
        monkeypatch.setenv(k, os.environ.get(k, ""))
    return monkeypatch


def test_paths_from_the_package_location(clean_env):
    assert paths.pyjetscape_dir() == PYJETSCAPE
    assert paths.examples_dir() == PYJETSCAPE / "example"
    root = PYJETSCAPE.parents[3]
    assert paths.xscape_root() == root
    assert paths.build_dir() == root / "build_gpu"
    assert paths.main_xml() == root / "config" / "jetscape_main.xml"
    assert paths.main_xml().exists()


def test_paths_from_the_environment(clean_env, tmp_path):
    clean_env.setenv("XSCAPE", str(tmp_path / "xs"))
    assert paths.xscape_root() == tmp_path / "xs"
    assert paths.build_dir() == tmp_path / "xs" / "build_gpu"
    assert paths.main_xml() == tmp_path / "xs" / "config" / "jetscape_main.xml"
    clean_env.setenv("JETSCAPE_BUILD", str(tmp_path / "b"))
    assert paths.build_dir() == tmp_path / "b"


def test_run_prod_re_exports_the_same_functions():
    spec = importlib.util.spec_from_file_location(
        "_run_prod_w", PYJETSCAPE / "example" / "prod_AuAu_0_10" / "run_prod.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    for name in ("enter_workdir", "job_workdir", "leave_workdir", "add_workdir_args",
                 "_absolute_parent_paths", "WORKDIR_LINKS", "SEED_MAX", "add_seed_args",
                 "seed_args_problems", "output_name", "seed_registry", "resolve_seed",
                 "seed_provenance", "pythia_data_problem"):
        assert getattr(rp, name) is getattr(workdir, name), name


def _stand_in(tmp_path):
    """A build tree with music_input and two of the linked dirs, and the two XMLs."""
    build = tmp_path / "xs" / "build_gpu"
    (build / "tables").mkdir(parents=True)
    (build / "LBT-tables").mkdir()
    (build / "music_input").write_text("music\n")
    main = tmp_path / "xs" / "config" / "jetscape_main.xml"
    main.parent.mkdir(parents=True)
    main.write_text("<jetscape><iSS><path>../external_packages/iSS/x</path></iSS>"
                    "<out>./FinalPartonsInfo.dat</out></jetscape>")
    user = tmp_path / "mine" / "my_job.xml"
    user.parent.mkdir()
    user.write_text("<jetscape><nEvents>3</nEvents>"
                    "<t>../external_packages/iSS/iSS_tables</t></jetscape>")
    return build, main, user


def test_jobdir_enters_and_cleans_up(clean_env, tmp_path):
    build, main, user = _stand_in(tmp_path)
    user_text = user.read_text()
    start = os.getcwd()
    out = tmp_path / "out"
    with JobDir(out, user, build=build, main_xml=main) as job:
        assert os.getcwd() == job.path == str(out / "work" / "my_job")
        assert job.user_xml == str(out / "my_job.xml")
        assert (Path(job.path) / "music_input").read_text() == "music\n"
        assert os.path.islink(Path(job.path) / "tables")
        assert os.path.islink(Path(job.path) / "LBT-tables")
        assert not os.path.lexists(Path(job.path) / "eps09")    # not in this build
        iss = str(tmp_path / "xs" / "external_packages" / "iSS")
        assert iss + "/iSS_tables" in Path(job.user_xml).read_text()
        main_text = Path(job.main_xml).read_text()
        assert iss + "/x" in main_text and "./FinalPartonsInfo.dat" in main_text
        assert os.environ["LBT_TABLES_PATH"] == str(build / "LBT-tables")
        Path("events_summary.dat").write_text("side file\n")
    assert os.getcwd() == start
    assert user.read_text() == user_text                        # the source is not touched
    assert (out / "my_job.xml").exists()
    assert not (out / "work").exists()


def test_jobdir_keeps_the_directory_after_an_exception(clean_env, tmp_path, capsys):
    build, main, user = _stand_in(tmp_path)
    start = os.getcwd()
    with pytest.raises(RuntimeError):
        with JobDir(tmp_path / "out", user, build=build, main_xml=main, name="j") as job:
            raise RuntimeError("event failed")
    assert os.getcwd() == start
    assert Path(job.path).is_dir() and "working directory kept" in capsys.readouterr().out
    with JobDir(tmp_path / "out2", user, build=build, main_xml=main, keep=True) as job:
        pass
    assert Path(job.path).is_dir()


def test_jobdir_uses_the_environment_and_reports_a_missing_build(clean_env, tmp_path):
    build, main, user = _stand_in(tmp_path)
    clean_env.setenv("XSCAPE", str(tmp_path / "xs"))
    job = JobDir(tmp_path / "out", user)
    assert job.build == str(build) and job.source_main_xml == str(main)
    clean_env.setenv("JETSCAPE_BUILD", str(tmp_path / "nowhere"))
    start = os.getcwd()
    with pytest.raises(FileNotFoundError, match="music_input"):
        with JobDir(tmp_path / "out", user):
            pass
    assert os.getcwd() == start and not (tmp_path / "out").exists()
