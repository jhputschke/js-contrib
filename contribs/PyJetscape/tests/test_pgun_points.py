"""
tests/test_pgun_points.py

run_prod_jet.py --hard pgun --pgun-grid / --pgun-y / --pgun-phi / --pgun-id: parsing the grid
and checking it against the other options (which also sets --reuse), and the XML it writes for
X-SCAPE's PGun (<points>, <rapidity>, <phi>, <parID>).  The last test needs the compiled
extension (PGUN_HAS_POINTS, pgun_points).

    pytest tests/test_pgun_points.py -q
"""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "example"


def _run_prod_jet():
    spec = importlib.util.spec_from_file_location(
        "_run_prod_jet_pgun", EXAMPLE / "prod_AuAu_0_10_jet" / "run_prod_jet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rpj = _run_prod_jet()


def _args(**kw):
    a = dict(hard="pgun", pgun_pt=60.0, pgun_y=None, pgun_phi=None, pgun_id=None,
             pgun_grid=None, pgun_per_point=1, reuse=1, events=6)
    a.update(kw)
    return SimpleNamespace(**a)


def test_grid_sets_points_and_reuse():
    a = _args(pgun_grid="20,50:0,1.5,3", pgun_per_point=2, events=24)
    rpj.check_pgun(a)
    # point k = iy * npT + ipT: event i uses point i mod K
    assert a.pgun_points == [(20, 0), (50, 0), (20, 1.5), (50, 1.5), (20, 3), (50, 3)]
    assert a.reuse == 12


def test_no_grid_leaves_reuse():
    a = _args(pgun_y=2.0, reuse=3)
    rpj.check_pgun(a)
    assert a.pgun_points is None and a.reuse == 3
    b = _args(hard="pythia", pgun_pt=60.0)
    rpj.check_pgun(b)                                   # the pgun defaults: nothing to check
    assert b.pgun_points is None


@pytest.mark.parametrize("kw", [
    dict(hard="pythia", pgun_y=1.0),                     # --pgun-* without --hard pgun
    dict(hard="pythia", pgun_grid="20:0"),
    dict(pgun_per_point=2),                              # without --pgun-grid
    dict(pgun_grid="20:0,1", pgun_y=1.0),                # the grid gives y
    dict(pgun_grid="20:0,1", reuse=2),                   # the grid sets reuse
    dict(pgun_grid="20:0,1", events=3),                  # not a multiple of 2 points
    dict(pgun_grid="20,0:0"),                            # pT must be > 0
    dict(pgun_grid="20;0"),
    dict(pgun_grid="a:0"),
    dict(pgun_grid=":0"),
    dict(pgun_grid="20:0", pgun_per_point=0),
])
def test_refuses(kw):
    with pytest.raises(ValueError):
        rpj.check_pgun(_args(**kw))


def test_xml_block():
    """The <Hard><PGun> block configure_xml writes (the same _set calls)."""
    a = _args(pgun_grid="20,50:0,3", pgun_per_point=1, events=4, pgun_id=2, pgun_phi=0.5)
    rpj.check_pgun(a)
    hard = ET.Element("Hard")
    pgun = ET.SubElement(hard, "PGun")
    rpj._set(pgun, "pT", a.pgun_pt)
    for tag, v in (("rapidity", a.pgun_y), ("phi", a.pgun_phi), ("parID", a.pgun_id)):
        if v is not None:
            rpj._set(pgun, tag, v)
    rpj._set(pgun, "points", " ".join(f"{pt:g} {y:g}" for pt, y in a.pgun_points))
    assert pgun.find("points").text.split() == "20 0 50 0 20 3 50 3".split()
    assert pgun.find("rapidity") is None
    assert float(pgun.find("phi").text) == 0.5 and int(pgun.find("parID").text) == 2


def test_main_xml_declares_the_tags():
    """X-SCAPE refuses user XML tags its main XML lacks: <rapidity>, <phi>, <points>."""
    main = Path(rpj.rp.MAIN_XML)
    if not main.exists():
        pytest.skip(f"no {main}")
    g = ET.parse(main).getroot().find("Hard/PGun")
    assert g is not None and all(g.find(t) is not None for t in ("rapidity", "phi", "points"))


def test_binding_flag():
    core = pytest.importorskip("jetscape.pyjetscape_core")
    assert core.PGUN_HAS_POINTS is True
    assert callable(core.pgun_points)
