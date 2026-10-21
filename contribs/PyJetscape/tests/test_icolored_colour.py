"""
tests/test_icolored_colour.py

iColoredHadronization's colour checks (docs/Plans/PLAN_lbt_colour_tags.md): a fragmentation
that Pythia gives up on throws instead of handing the input partons on as hadrons, and the
colour audit is bound.  Runs the compiled module on hand-made parton lists through
hadronize_input (no beam remnants, no hydro).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

core = pytest.importorskip("jetscape.pyjetscape_core")
if not getattr(core, "ICOLORED_HAS_COLOUR_AUDIT", False):
    pytest.skip("X-SCAPE without iColoredHadronization's colour audit",
                allow_module_level=True)

PROD = ROOT / "example" / "prod_AuAu_0_10"
_spec = importlib.util.spec_from_file_location("_run_prod", PROD / "run_prod.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

AUDIT_KEYS = {"partons", "remnants", "holes", "hole_energy", "skipped", "unpaired_col",
              "unpaired_acol", "duplicate_tags", "relabelled_tags"}


@pytest.fixture(scope="module")
def icolored(tmp_path_factory):
    work = tmp_path_factory.mktemp("icolored")
    user = ET.fromstring(
        "<jetscape><enableAutomaticTaskListDetermination>false"
        "</enableAutomaticTaskListDetermination><nEvents>1</nEvents><vlevel>0</vlevel>"
        "<Random><seed>1</seed></Random>"
        "<JetHadronization><name>iColored</name><eCMforHadronization>200"
        "</eCMforHadronization><weak_decays>off</weak_decays></JetHadronization></jetscape>")
    for w in ("JetScapeWriterAscii", "JetScapeWriterAsciiGZ", "JetScapeWriterHepMC",
              "JetScapeWriterFinalStatePartonsAscii", "JetScapeWriterFinalStateHadronsAscii"):
        ET.SubElement(user, w).text = "off"
    user_xml = work / "user.xml"
    ET.ElementTree(user).write(user_xml)
    main_xml = work / "jetscape_main.xml"
    rp._absolute_parent_paths(os.path.join(rp.XSCAPE, "config", "jetscape_main.xml"),
                              str(main_xml), os.path.join(rp.XSCAPE, "build_gpu"))
    cwd = os.getcwd()
    os.chdir(work)
    try:
        js = core.JetScapePerEvent()
        js.SetXMLMainFileName(str(main_xml))
        js.SetXMLUserFileName(str(user_xml))
        js.Init()
        module = core.create_module("iColoredHadronization")
        module.Init()
        yield module
    finally:
        os.chdir(cwd)


def _input(rows):
    """rows of (pid, col, acol, E, px, py, pz) -> hadronize_input's arrays (massless)."""
    r = np.asarray(rows, dtype=np.float64)
    return dict(pid=r[:, 0].astype(np.int32), col=r[:, 1].astype(np.int32),
                acol=r[:, 2].astype(np.int32), p=np.ascontiguousarray(r[:, 3:7]),
                mass=np.zeros(len(r)))


def test_audit_is_bound(icolored):
    audit = core.jet_hadronization_colour_audit(icolored)
    assert set(audit) == AUDIT_KEYS
    assert all(v == 0 for v in audit.values())      # nothing hadronized in an event yet


def test_a_colour_singlet_gives_hadrons(icolored):
    # a 20 GeV q qbar string along x
    out = core.hadronize_input(icolored, **_input([(2, 101, 0, 10.0, 10.0, 0.0, 0.0),
                                                   (-2, 0, 101, 10.0, -10.0, 0.0, 0.0)]),
                               seed=7)
    pid = np.abs(np.asarray(out["pid"]))
    assert len(pid) > 2
    assert not np.any((pid <= 6) | (pid == 21))      # no parton is passed on as a hadron
    assert np.sum(np.asarray(out["p"])[:, 0]) == pytest.approx(20.0, rel=1e-6)


def test_a_failed_fragmentation_throws(icolored):
    # a lone quark: its colour line ends nowhere, Pythia gives up on the event.  Before the
    # check, the quark itself came back as a "hadron".
    with pytest.raises(RuntimeError, match="could not hadronize"):
        core.hadronize_input(icolored, **_input([(2, 101, 0, 10.0, 10.0, 0.0, 0.0)]), seed=7)
