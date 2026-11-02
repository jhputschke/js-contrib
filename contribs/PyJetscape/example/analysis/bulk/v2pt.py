"""v2(pT) of charged hadrons from extra_hard_hists.py's per-event sums, and the published
data the notebooks compare with (HEPData, ./get_hepdata.sh).

    import v2pt
    est = v2pt.estimator(d, "full", "gap1")      # d: the npz of extra_hard_hists.py
    val = est.f(*[a.mean(0) for a in est.arrays])
    alice = v2pt.alice("5-10")                    # HEPData table: lo, hi, x, value, stat, sys

Estimators (extra_hard_hists.py's docstring has the definitions):

    gap1    v2{2, |Delta eta| > 1} as ALICE (arXiv:1804.02944): <<2'>> / sqrt(<<2>>)
    gap15   the same with |Delta eta| > 1.5 (POIs |eta| < 1, reference 1.5 < |eta| < 2.6)
    psi     <cos 2(phi - Psi_2,bg)> / R, Psi_2 of the jet-free background from all its
            samples; R the full-plane resolution from the two halves (``resolution``)

Each estimator is a function of means over events, so the notebooks' jackknife (over the
hydro backgrounds) applies unchanged: ``est.f(*means)``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HEPDATA = os.path.join(HERE, "..", "runs", "hepdata")
METHODS = ("gap1", "gap15", "psi")
LABEL = {"gap1": r"$v_2\{2, |\Delta\eta| > 1\}$",
         "gap15": r"$v_2\{2, |\Delta\eta| > 1.5\}$",
         "psi": r"$v_2\{\Psi_{2,\mathrm{bg}}\}$"}

#: HEPData tables: (record, file) per centrality class
ALICE_TABLES = {"0-5": ("ins1666817", "v22Deta1_502TeVdif_cen0.yaml"),
                "5-10": ("ins1666817", "v22Deta1_502TeVdif_cen1.yaml")}
CMS_TABLES = {"0-5": ("ins1511868", "data1.yaml"), "5-10": ("ins1511868", "data2.yaml")}


def hepdata(record, table, version=1, base=None):
    """A HEPData YAML table -> dict lo, hi (bin edges; NaN where only a value is given), x
    (bin centre or the given value), value, stat, sys (symmetric errors)."""
    import yaml
    path = os.path.join(base or HEPDATA, f"HEPData-{record}-v{version}-yaml", table)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run analysis/bulk/get_hepdata.sh first")
    with open(path) as f:
        t = yaml.safe_load(f)
    xs = t["independent_variables"][0]["values"]
    ys = t["dependent_variables"][0]["values"]
    lo = np.array([x.get("low", np.nan) for x in xs], float)
    hi = np.array([x.get("high", np.nan) for x in xs], float)
    x = np.array([x.get("value", np.nan) for x in xs], float)
    x = np.where(np.isfinite(x), x, 0.5 * (lo + hi))
    err = {"stat": [], "sys": []}
    for y in ys:
        e = {k["label"]: float(k["symerror"]) for k in y.get("errors", [])}
        for k in err:
            err[k].append(e.get(k, 0.0))
    return {"lo": lo, "hi": hi, "x": x, "value": np.array([y["value"] for y in ys], float),
            "stat": np.array(err["stat"]), "sys": np.array(err["sys"]),
            "record": record, "table": table}


def alice(cls):
    """ALICE v2{2, |Delta eta| > 1}(pT), Pb-Pb 5.02 TeV, |eta| < 0.8, class "0-5" or "5-10"."""
    return hepdata(*ALICE_TABLES[cls])


def cms(cls):
    """CMS v2{SP}(pT) to 100 GeV, Pb-Pb 5.02 TeV, |eta| < 1, HF reference (|Delta eta| > 3)."""
    return hepdata(*CMS_TABLES[cls])


def _r(chi):
    """The event-plane resolution <cos 2(Psi_est - Psi)> for resolution parameter chi."""
    from scipy.special import ive
    x = 0.5 * chi ** 2
    return np.sqrt(np.pi) / 2 * chi * (ive(0, x) + ive(1, x))


def resolution(cos_sub):
    """The full-plane resolution from <cos 2(Psi_A - Psi_B)> of two equal halves: R_sub =
    sqrt(<cos>), chi_sub from R(chi_sub) = R_sub, R_full = R(sqrt(2) chi_sub).  NaN for a
    non-positive <cos>."""
    from scipy.optimize import brentq
    c = float(cos_sub)
    if not np.isfinite(c) or c <= 0:
        return np.nan
    r_sub = np.sqrt(min(c, 1.0))
    if r_sub >= _r(40.0):
        return 1.0
    chi = brentq(lambda x: _r(x) - r_sub, 1e-6, 40.0)
    return float(_r(np.sqrt(2.0) * chi))


@dataclass
class Estimator:
    """``f(*means)`` -> v2 per bin, ``arrays`` the per-event inputs (N, ...)."""
    f: Callable
    arrays: tuple


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def estimator(d, comp, method):
    """v2(pT) of composition ``comp`` ("bg", "jet", "full") with ``method`` (METHODS) from
    the npz of extra_hard_hists.py."""
    k = f"{comp}.v2_{method}"
    if method == "psi":
        def f(num, den, res):
            return _div(num, den) / resolution(res)
        return Estimator(f, (d[f"{k}_num"], d[f"{k}_den"], d["psi2_res"]))

    def f(num, den, rnum, rden):
        ref = _div(np.atleast_1d(rnum), np.atleast_1d(rden))[0]
        return _div(num, den) / (np.sqrt(ref) if ref > 0 else np.nan)
    return Estimator(f, (d[f"{k}_num"], d[f"{k}_den"], d[f"{k}_rnum"], d[f"{k}_rden"]))


def poi_counts(d, comp, mask=None):
    """Charged hadrons per v2 bin at |eta| < 0.8: per sample (the mean over an event's
    samples), summed over the selected events.  How many distinct hadrons a bin's v2 rests
    on: an event's oversamples share one fluid and its fragments repeat with k mod n_frag, so
    multiplying by the samples would overstate it."""
    m = slice(None) if mask is None else mask
    return d[f"{comp}.v2_psi_den"][m].sum(0)


def ncoll_classes(n_coll):
    """The 0-10 % sample split at the median N_coll: "0-5" (the larger half) and "5-10".
    A proxy: the experiments' classes are multiplicity (or energy) percentiles."""
    med = np.median(n_coll)
    return {"0-5": n_coll >= med, "5-10": n_coll < med}
