"""
tests/test_edge_threshold.py

run_prod_jet.py --edge-e-threshold: the default, MUSIC's freeze-out energy density from the
job XML (eps_switch, or e(freezeout_temperature) from an EOS 9/91 table in the build).
Nothing here needs the compiled extension or a build: the table is a synthetic one.

    pytest tests/test_edge_threshold.py -q
"""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "example"


def _run_prod_jet():
    spec = importlib.util.spec_from_file_location(
        "_run_prod_jet_edge_t", EXAMPLE / "prod_AuAu_0_10_jet" / "run_prod_jet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rpj = _run_prod_jet()

MAIN = """<jetscape><Hydro><MUSIC>
  <EOS>91</EOS><use_eps_for_freeze_out>0</use_eps_for_freeze_out>
  <freezeout_temperature>0.136</freezeout_temperature><eps_switch>0.3</eps_switch>
</MUSIC></Hydro></jetscape>"""


def _setup(tmp_path, user_music):
    main = tmp_path / "main.xml"
    main.write_text(MAIN)
    build = tmp_path / "build"
    (build / "EOS" / "hotQCD").mkdir(parents=True)
    # rows (e, P, s, T): e = 10 T over T in [0.05, 0.5]
    T = np.linspace(0.05, 0.5, 46)
    for name in rpj.EOS_TABLES.values():
        np.stack([10 * T, T, T, T], axis=1).astype("<f8").tofile(
            build / "EOS" / "hotQCD" / name)
    root = ET.fromstring(f"<jetscape><Hydro><MUSIC>{user_music}</MUSIC></Hydro></jetscape>")
    return root, str(main), str(build)


def test_e_of_T_fo_from_the_eos_table(tmp_path):
    root, main, build = _setup(tmp_path, "<EOS>9</EOS><freezeout_temperature>0.15"
                                         "</freezeout_temperature>")
    e, src = rpj.freezeout_e(root, main, build)
    assert e == pytest.approx(1.5) and "EOS 9" in src


def test_missing_tags_come_from_the_main_xml(tmp_path):
    root, main, build = _setup(tmp_path, "")
    e, src = rpj.freezeout_e(root, main, build)       # EOS 91, T_fo 0.136
    assert e == pytest.approx(1.36) and "EOS 91" in src


def test_eps_switch_when_freezing_out_on_energy_density(tmp_path):
    root, main, build = _setup(tmp_path, "<use_eps_for_freeze_out>1</use_eps_for_freeze_out>"
                                         "<eps_switch>0.12</eps_switch>")
    assert rpj.freezeout_e(root, main, build) == (0.12, "eps_switch")


def test_no_threshold_for_an_eos_without_a_table(tmp_path):
    root, main, build = _setup(tmp_path, "<EOS>20</EOS>")
    e, src = rpj.freezeout_e(root, main, build)
    assert e is None and "--edge-e-threshold" in src


def test_the_production_xml_gives_musics_value():
    """The production XML (EOS 9, T_fo 0.15) against the real table, if this checkout has
    a build: 0.2342 GeV/fm^3, the e MUSIC's surfaces carry."""
    build = Path(rpj.rp.XSCAPE) / "build_gpu"
    if not (build / "EOS" / "hotQCD" / rpj.EOS_TABLES["9"]).exists():
        pytest.skip("no build_gpu EOS table")
    root = ET.parse(rpj.USER_XML).getroot()
    e, _ = rpj.freezeout_e(root, str(Path(rpj.rp.XSCAPE) / "config" / "jetscape_main.xml"),
                           str(build))
    assert e == pytest.approx(0.2342, abs=1e-4)
