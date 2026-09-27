"""
tests/test_pthat_bins.py

run_prod_jet.py --pthat-bins: parsing the windows and checking them against the other
options (which also sets --reuse).  Nothing here needs the compiled extension or a build.

    pytest tests/test_pthat_bins.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "example"


def _run_prod_jet():
    spec = importlib.util.spec_from_file_location(
        "_run_prod_jet_t", EXAMPLE / "prod_AuAu_0_10_jet" / "run_prod_jet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rpj = _run_prod_jet()


def _args(**kw):
    a = dict(pthat_bins=None, jets_per_bin=1, hard="pythia", pthat_min=None, pthat_max=None,
             reuse=1, events=6)
    a.update(kw)
    return SimpleNamespace(**a)


def test_parse_pthat_bins():
    assert rpj.parse_pthat_bins("20-40,50-70") == [(20.0, 40.0), (50.0, 70.0)]
    assert rpj.parse_pthat_bins(" 0-10 , 10.5-20 ,") == [(0.0, 10.0), (10.5, 20.0)]
    for bad in ("", "20", "40-20", "20-40-60", "a-b", "20.25-40", "-5-10"):
        with pytest.raises(ValueError):
            rpj.parse_pthat_bins(bad)
    assert rpj.pthat_bins_text([(20.0, 40.0), (50.5, 70.0)]) == "20 40 50.5 70"


def test_check_pthat_bins_sets_reuse():
    a = _args(pthat_bins="20-40,50-70,70-90", jets_per_bin=2, events=12)
    rpj.check_pthat_bins(a)
    assert a.reuse == 6 and a.pthat_windows == [(20, 40), (50, 70), (70, 90)]
    b = _args()
    rpj.check_pthat_bins(b)                          # no windows: nothing changes
    assert b.reuse == 1 and b.pthat_windows is None


@pytest.mark.parametrize("kw", [
    dict(jets_per_bin=2),                                        # without --pthat-bins
    dict(pthat_bins="20-40,50-70", hard="pgun"),
    dict(pthat_bins="20-40,50-70", pthat_min=10.0),
    dict(pthat_bins="20-40,50-70", reuse=2),                     # --reuse is set from it
    dict(pthat_bins="20-40,50-70", jets_per_bin=0),
    dict(pthat_bins="20-40,50-70", events=5),                    # not a multiple of 2
])
def test_check_pthat_bins_refuses(kw):
    with pytest.raises(ValueError):
        rpj.check_pthat_bins(_args(**kw))
