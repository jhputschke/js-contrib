"""
tests/test_icolored_colour.py

iColoredHadronization's colour checks and closure (docs/Plans/PLAN_lbt_colour_tags.md): a
fragmentation that Pythia gives up on throws instead of handing the input partons on as
hadrons; duplicate tags fail the event; the colour lines that scatterings off the medium
leave open are closed as <colour_closure> says, and the audit counts it.  Runs the compiled
module on hand-made parton lists through hadronize_input / hadronize_partons (no beam
remnants, no hydro).
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
              "unpaired_acol", "duplicate_tags", "relabelled_tags", "reconnected",
              "reconnected_cross", "partners", "partner_energy", "fakes", "fake_energy",
              "retries"}


@pytest.fixture(scope="module")
def icolored(tmp_path_factory):
    work = tmp_path_factory.mktemp("icolored")
    user = ET.fromstring(
        "<jetscape><enableAutomaticTaskListDetermination>false"
        "</enableAutomaticTaskListDetermination><nEvents>1</nEvents><vlevel>0</vlevel>"
        "<Random><seed>1</seed></Random>"
        "<JetHadronization><name>iColored</name><eCMforHadronization>200"
        "</eCMforHadronization><weak_decays>off</weak_decays><store_input>1</store_input>"
        "</JetHadronization></jetscape>")
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


# ---------------------------------------------------------------------------- the closure

def _partons(rows):
    """rows of (shower, pid, pstat, col, acol, E, px, py, pz) -> FINAL_PARTON_COLUMNS."""
    out = np.zeros((len(rows), 14))
    for k, (sh, pid, st, c, a, e, px, py, pz) in enumerate(rows):
        out[k, [0, 1, 2, 3, 4, 5, 6, 12, 13]] = [sh, pid, st, e, px, py, pz, c, a]
    return out


def _hadronize(module, mode, rows, seed=11):
    assert core.jet_hadronization_set_colour_closure(module, mode) == mode
    out = core.hadronize_partons(module, _partons(rows), seed=seed)
    audit = core.jet_hadronization_colour_audit(module)
    given = core.hadronization_input_numpy(module)
    return out, audit, given


def _check_hadrons(out, given):
    pid = np.abs(np.asarray(out["pid"]))
    assert len(pid) > 0 and not np.any((pid <= 6) | (pid == 21))
    # Pythia's strings conserve the four-momentum of what they were given (decays off)
    assert np.sum(np.asarray(out["p"]), axis=0) == pytest.approx(
        np.sum(given["p"], axis=0), rel=1e-5, abs=1e-4)   # hadrons are float32


def _paired(given):
    c, a = given["col"], given["acol"]
    return sorted(c[c != 0].tolist()) == sorted(a[a != 0].tolist()) and \
        len(set(c[c != 0].tolist())) == int(np.sum(c != 0))


# A q qbar dijet (showers 0 and 1, colour line 101), the quark scattered off a thermal gluon
# (q g -> q g, Matter's flow): jet-out (201, 0), recoil gluon (202, 201), hole = the
# incoming thermal gluon (202, 101). Open: the recoil's colour 202 (shower 0) and the
# antiquark's anticolour 101 (shower 1): one line ends in the medium.
QG = [(0, 2, 0, 201, 0, 30.0, 30.0, 0.0, 0.0),
      (0, 21, 1, 202, 201, 3.0, 2.0, 2.0, 1.0),
      (0, 21, -1, 202, 101, 1.0, 0.0, 1.0, 0.0),
      (1, -2, 0, 0, 101, 32.0, -32.0, 0.0, 0.0)]

# The quark scattered off a thermal quark of another flavour (q q' -> q q'): jet-out (201, 0),
# recoil (101, 0), hole (201, 0). Open: the jet-out's colour 201 (a triplet left the medium).
QQ = [(0, 2, 0, 201, 0, 30.0, 30.0, 0.0, 0.0),
      (0, 1, 1, 101, 0, 3.0, 2.0, 2.0, 1.0),
      (0, 1, -1, 201, 0, 1.0, 0.0, 1.0, 0.0),
      (1, -2, 0, 0, 101, 32.0, -32.0, 0.0, 0.0)]

# Two scatterings in shower 0, off a thermal quark and off a thermal antiquark: open colour
# 201 and open anticolour 301 in the same shower.
TWO = [(0, 2, 0, 201, 0, 30.0, 30.0, 0.0, 0.0),
       (0, 2, 1, 101, 0, 3.0, 2.0, 2.0, 1.0),
       (0, 2, -1, 201, 0, 1.0, 0.0, 1.0, 0.0),
       (0, -1, 0, 0, 301, 20.0, 18.0, 8.0, 3.0),
       (0, -1, 1, 0, 102, 2.0, 1.0, 1.5, 0.8),
       (0, -1, -1, 0, 301, 1.0, 0.0, 0.0, 1.0),
       (1, -2, 0, 0, 101, 32.0, -32.0, 0.0, 0.0),
       (1, 1, 0, 102, 0, 22.0, -20.0, -9.0, 0.0)]


def test_none_throws_on_an_open_line(icolored):
    with pytest.raises(RuntimeError, match="Unpaired colors"):
        _hadronize(icolored, "none", QG)
    audit = core.jet_hadronization_colour_audit(icolored)
    assert (audit["unpaired_col"], audit["unpaired_acol"], audit["holes"]) == (1, 1, 1)


def test_reconnect_inserts_the_gluon_partner_across_showers(icolored):
    out, audit, given = _hadronize(icolored, "reconnect", QG)
    # the open ends are in different showers: no join within a shower, the hole's medium
    # partner (a gluon with the hole's tags swapped) closes both
    assert (audit["reconnected"], audit["partners"], audit["fakes"]) == (0, 1, 0)
    assert audit["hole_energy"] == pytest.approx(1.0)
    assert audit["partner_energy"] == pytest.approx(1.0)
    assert given["pid"][-1] == 21 and (given["col"][-1], given["acol"][-1]) == (101, 202)
    assert _paired(given)
    _check_hadrons(out, given)


def test_reconnect_global_joins_instead(icolored):
    out, audit, given = _hadronize(icolored, "reconnect_global", QG)
    assert (audit["reconnected"], audit["partners"], audit["fakes"]) == (1, 0, 0)
    assert len(given["pid"]) == 3 and _paired(given)          # no parton added
    _check_hadrons(out, given)


def test_quark_hole_gets_an_antiquark_partner(icolored):
    out, audit, given = _hadronize(icolored, "reconnect", QQ)
    assert (audit["unpaired_col"], audit["unpaired_acol"]) == (1, 0)
    assert (audit["reconnected"], audit["partners"]) == (0, 1)
    assert given["pid"][-1] == -1 and given["acol"][-1] == 201
    assert _paired(given)
    _check_hadrons(out, given)


def test_reconnect_joins_within_a_shower_before_partners(icolored):
    out, audit, given = _hadronize(icolored, "reconnect", TWO)
    assert (audit["holes"], audit["unpaired_col"], audit["unpaired_acol"]) == (2, 1, 1)
    assert (audit["reconnected"], audit["partners"], audit["fakes"]) == (1, 0, 0)
    assert _paired(given)
    _check_hadrons(out, given)
    # thermal: both holes get their partner, nothing is joined
    out, audit, given = _hadronize(icolored, "thermal", TWO)
    assert (audit["reconnected"], audit["partners"]) == (0, 2)
    assert audit["partner_energy"] == pytest.approx(2.0)
    assert sorted(given["pid"][-2:].tolist()) == [-2, 1]
    _check_hadrons(out, given)


def test_fake_closes_with_soft_beam_partons(icolored):
    out, audit, given = _hadronize(icolored, "fake", QG)
    assert (audit["partners"], audit["fakes"]) == (0, 2)
    assert audit["fake_energy"] == pytest.approx(2.0)
    assert _paired(given)
    _check_hadrons(out, given)


def test_no_open_line_no_change(icolored):
    rows = [r for r in QG if r[2] >= 0][:1] + [(1, -2, 0, 0, 201, 32.0, -32.0, 0.0, 0.0)]
    out, audit, given = _hadronize(icolored, "reconnect", rows)
    assert (audit["unpaired_col"], audit["reconnected"], audit["partners"],
            audit["fakes"]) == (0, 0, 0, 0)
    assert len(given["pid"]) == 2
    _check_hadrons(out, given)


def test_duplicate_tags_fail_the_event(icolored):
    rows = [(0, 2, 0, 101, 0, 10.0, 10.0, 0.0, 0.0), (0, -2, 0, 0, 101, 10.0, -10.0, 0.0, 0.0),
            (1, 2, 0, 101, 0, 10.0, 0.0, 10.0, 0.0), (1, -2, 0, 0, 101, 10.0, 0.0, -10.0, 0.0)]
    with pytest.raises(RuntimeError, match="duplicate colour tags"):
        _hadronize(icolored, "reconnect", rows)
    assert core.jet_hadronization_colour_audit(icolored)["duplicate_tags"] == 2


def test_liquefied_partons_are_skipped(icolored):
    rows = QQ + [(0, 21, -11, 5, 6, 2.0, 1.0, 1.0, 1.0)]
    out, audit, given = _hadronize(icolored, "reconnect", rows)
    assert audit["skipped"] == 1 and audit["partners"] == 1
    _check_hadrons(out, given)


def test_remnants_go_to_pythia_whatever_their_status(icolored):
    # beam remnants (shower -1) carry iMATTER's negative statuses: never skipped
    rows = [(0, 2, 1000, 101, 0, 10.0, 10.0, 0.0, 0.0), (-1, -2, -1, 0, 101, 10.0, 0.0, 0.0, 10.0)]
    out, audit, given = _hadronize(icolored, "reconnect", rows)
    assert (audit["partons"], audit["remnants"], audit["holes"], audit["skipped"]) == (1, 1, 0, 0)
    assert len(given["pid"]) == 2
    _check_hadrons(out, given)


def test_totals_add_up_the_audits(icolored):
    before = core.jet_hadronization_colour_totals(icolored)
    _hadronize(icolored, "reconnect", QG)
    with pytest.raises(RuntimeError):
        _hadronize(icolored, "none", QG)
    after = core.jet_hadronization_colour_totals(icolored)
    assert after["events"] == before["events"] + 1
    assert after["failed"] == before["failed"] + 1
    assert after["holes"] == before["holes"] + 1
    assert after["partners"] == before["partners"] + 1
    assert after["open_ends"] == before["open_ends"] + 2
