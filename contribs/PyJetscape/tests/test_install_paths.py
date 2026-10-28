"""
tests/test_install_paths.py

An installed X-SCAPE (the xscape wheel, docs: "Packaging X-SCAPE for Metal Macs"):
jetscape.paths falls back to the wheel's installed tree, '../' paths with no target to the
build's asset of the same name, the hydro label from the core's MUSIC_BACKEND, the iSS
commit from BUILD_INFO.txt, and xscape-examples.  A stand-in installed tree in a temporary
directory; nothing here runs X-SCAPE.

    pytest tests/test_install_paths.py -q
"""

from __future__ import annotations

import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from jetscape import copy_examples, hadronizer, paths, workdir


@pytest.fixture
def installed(monkeypatch, tmp_path):
    """A stand-in xscape wheel (site/xscape/share/xscape) and a jetscape package that is
    not inside a checkout; no XSCAPE/JETSCAPE_BUILD from the caller."""
    for k in ("XSCAPE", "JETSCAPE_BUILD"):
        monkeypatch.delenv(k, raising=False)
    pkg = tmp_path / "site" / "xscape"
    tree = pkg / "share" / "xscape"
    (tree / "config").mkdir(parents=True)
    (tree / "config" / "jetscape_main.xml").write_text("<jetscape/>")
    (tree / "music_input").write_text("")
    (tree / "iSS_tables").mkdir()
    (tree / "iSS_parameters.dat").write_text("")
    (tree / "examples" / "pythonic").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    monkeypatch.delitem(sys.modules, "xscape", raising=False)
    monkeypatch.setattr(paths, "pyjetscape_dir", lambda: tmp_path / "site" / "PyJetscape")
    return tree


def test_installed_tree_is_root_and_build(installed):
    assert paths.installed_tree() == installed
    assert paths.xscape_root() == installed
    assert paths.build_dir() == installed
    assert paths.main_xml() == installed / "config" / "jetscape_main.xml"
    assert paths.examples_dir() == installed / "examples"


def test_environment_still_wins(installed, monkeypatch, tmp_path):
    monkeypatch.setenv("XSCAPE", str(tmp_path / "xs"))
    assert paths.xscape_root() == tmp_path / "xs"
    assert paths.build_dir() == tmp_path / "xs" / "build_gpu"
    monkeypatch.setenv("JETSCAPE_BUILD", str(tmp_path / "b"))
    assert paths.build_dir() == tmp_path / "b"


def test_no_tree_is_an_error(monkeypatch, tmp_path):
    for k in ("XSCAPE", "JETSCAPE_BUILD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(paths, "pyjetscape_dir", lambda: tmp_path / "a" / "b" / "c" / "d" / "e")
    monkeypatch.setattr(paths, "installed_tree", lambda: None)
    with pytest.raises(FileNotFoundError, match="XSCAPE"):
        paths.xscape_root()


def test_parent_paths_fall_back_to_the_build(tmp_path):
    build = tmp_path / "share" / "xscape"
    (build / "iSS_tables").mkdir(parents=True)
    (tmp_path / "share" / "present").mkdir()
    src = tmp_path / "main.xml"
    src.write_text("<a><t>../external_packages/iSS/iSS_tables</t><p>../present</p>"
                   "<m>../external_packages/nothing</m><o>./out.dat</o></a>")
    dst = tmp_path / "job.xml"
    workdir._absolute_parent_paths(str(src), str(dst), str(build))
    root = ET.parse(dst).getroot()
    assert root.find("t").text == str(build / "iSS_tables")              # fallback
    assert root.find("p").text == str(tmp_path / "share" / "present")     # target exists
    assert root.find("m").text == str(tmp_path / "share" / "external_packages" / "nothing")
    # only iSS's assets: another name with no target stays, even if the build has it
    (build / "MUSIC_input").write_text("")
    src.write_text("<a><h>../examples/test_hydro_files/MUSIC_input</h></a>")
    workdir._absolute_parent_paths(str(src), str(dst), str(build))
    assert ET.parse(dst).getroot().find("h").text == str(
        tmp_path / "share" / "examples" / "test_hydro_files" / "MUSIC_input")
    assert root.find("o").text == "./out.dat"


def test_hydro_label(monkeypatch):
    for backend, label in (("music4gpu-metal", "MUSIC (music4gpu)"),
                           ("music4gpu-cuda", "MUSIC (music4gpu)"), ("music", "MUSIC")):
        core = types.SimpleNamespace(MUSIC_BACKEND=backend)
        monkeypatch.setitem(sys.modules, "jetscape.pyjetscape_core", core)
        assert workdir.hydro_label("/x/build_gpu") == label
        assert workdir.hydro_label("/x/share/xscape") == label
    # a core built before MUSIC_BACKEND: the folder name, as before
    monkeypatch.setitem(sys.modules, "jetscape.pyjetscape_core", types.SimpleNamespace())
    assert workdir.hydro_label("/x/build_gpu") == "MUSIC (music4gpu)"
    assert workdir.hydro_label("/x/build") == "MUSIC"


def test_iss_commit_from_build_info(tmp_path):
    assert hadronizer.iss_commit(str(tmp_path)) == ""
    (tmp_path / "BUILD_INFO.txt").write_text(
        "# build\n. 1111 0 modified\nexternal_packages/iSS 6ac0333abc 0 modified\n")
    assert hadronizer.iss_commit(str(tmp_path)) == "6ac0333abc"


def test_copy_examples(monkeypatch, tmp_path):
    src = tmp_path / "example"
    (src / "pythonic" / "out_oo").mkdir(parents=True)
    (src / "pythonic" / "oo_workflow.py").write_text("")
    (src / "pythonic" / "out_oo" / "x.h5").write_text("")
    (src / "pythonic" / "run.h5").write_text("")
    monkeypatch.setattr(paths, "examples_dir", lambda: src)
    dest = tmp_path / "copy"
    assert copy_examples.main([str(dest)]) == 0
    assert (dest / "pythonic" / "oo_workflow.py").is_file()
    assert not (dest / "pythonic" / "out_oo").exists()
    assert not (dest / "pythonic" / "run.h5").exists()
    assert copy_examples.main([str(dest)]) == 1          # never overwrites


def test_hadronize_xml_iss_paths(tmp_path):
    xml = "<jetscape><nEvents>1</nEvents><Random><seed>1</seed></Random><SoftParticlization>" \
          "<iSS/></SoftParticlization></jetscape>"
    for tree, iss in ((tmp_path / "checkout", tmp_path / "checkout" / "external_packages" / "iSS"),
                      (tmp_path / "installed", tmp_path / "installed")):
        iss.mkdir(parents=True)
        (iss / "iSS_parameters.dat").write_text("")
        root = hadronizer.configure_hadronize_xml(ET.fromstring(xml), xscape=str(tree), seed=1,
                                                  n_events=1, oversample=1, keep_primaries=0)
        assert root.find("SoftParticlization/iSS/iSS_input_file").text == str(
            iss / "iSS_parameters.dat")
        assert root.find("SoftParticlization/iSS/iSS_table_path").text == str(iss / "iSS_tables")


def test_analysis_extra_is_the_analysis_env_list():
    import tomllib
    here = Path(__file__).resolve().parents[1]
    extra = tomllib.loads((here / "pyproject.toml").read_text())["project"][
        "optional-dependencies"]["analysis"]
    req = here.parents[1] / "utils" / "analysis_env" / "requirements.txt"
    listed = [line.split("#")[0].strip() for line in req.read_text().splitlines()]
    assert extra == [x for x in listed if x]


def test_copy_examples_runs_git_only_in_a_checkout(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(copy_examples.subprocess, "run",
                        lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(OSError()))
    assert copy_examples._tracked_files(str(tmp_path)) is None
    assert calls == []
