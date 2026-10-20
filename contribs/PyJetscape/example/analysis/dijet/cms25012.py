"""CMS HIN-25-012 (arXiv:2602.19431), HEPData ins3123406 (doi:10.17182/hepdata.168219; fetch
with ../zjet/get_hepdata.sh 25012 into ../runs/hepdata/): the eta-separated dijet-hadron
correlations, near side, R_asym - R_sym against Delta eta_ch,jet1.

    Table 1 (fig1_0)  R_asym(1.0-1.5) - R_sym, 1 < p_T^ch < 2 GeV: PbPb 0-30, 30-50, 50-80 %, pp
    Table 2 (fig1_1)  the same at 2-4 GeV
    Table 3 (fig2)    PbPb 0-30 % - pp: 1-2 GeV for R_asym 0.5-1.0, 1.0-1.5, 1.5-2.0; 2-4 GeV
                      for 1.0-1.5
    Table 4 (fig3)    PbPb - pp integrated over -2.4 < Delta eta < -0.6 against centrality
                      (50-80, 30-50, 0-30 %), 1-2 and 2-4 GeV, R_asym 1.0-1.5 and 1.5-2.0

Delta eta in 23 bins of 0.2 from -2.5 to 2.1 (edges at odd tenths).  The record gives no unit.
Table 4 is exactly the sum of Table 3's values over the 10 bins centred -2.4 ... -0.6, with the
statistical and the systematic errors each added in quadrature (checked on all four 0-30 %
entries): the self-consistent reading is that Tables 1-3 are yields per bin of 0.2 (``NORMS``
"per-bin", the default of deta.py --cms-norm), not per unit Delta eta ("per-unit": then
Table 4 would be 5x the integral).  The model's yields per unit Delta y go into CMS's
convention with :func:`to_cms`."""
import os

import numpy as np
import yaml

#: the model (per unit Delta y) -> CMS's tables: the factor per 0.2 bin
NORMS = {"per-bin": 0.2, "per-unit": 1.0}
PT = {"1-2": "1 < p", "2-4": "2 < p"}
GAPS = {"0.5-1.0": "(0.5, 1.0)", "1.0-1.5": "(1.0, 1.5)", "1.5-2.0": "(1.5, 2.0)"}


def _load(hep, name):
    with open(os.path.join(hep, f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def _quals(dv):
    return {q["name"]: str(q["value"]) for q in dv.get("qualifiers", [])}


def _series(d, dv):
    x = d["independent_variables"][0]["values"]
    lo = np.array([b["low"] for b in x], float)
    hi = np.array([b["high"] for b in x], float)
    vals = dv["values"]
    value = np.array([float(b["value"]) for b in vals])
    stat = np.array([float(b["errors"][0]["symerror"]) for b in vals])
    syst = np.array([float(b["errors"][1]["symerror"]) for b in vals])
    return {"lo": lo, "hi": hi, "x": 0.5 * (lo + hi), "value": value, "stat": stat,
            "syst": syst, "total": np.hypot(stat, syst)}


def _pt_of(q):
    s = " ".join(q.values())
    for k, tag in PT.items():
        if tag in s:
            return k
    raise ValueError(f"no p_T class in {q}")


def _gap_of(q):
    s = " ".join(q.values())
    for k, tag in GAPS.items():
        if tag in s:
            return k
    raise ValueError(f"no gap class in {q}")


def table12(hep):
    """Tables 1-2 -> {(pt, system): series}, pt '1-2' / '2-4', system '0-30', '30-50',
    '50-80' (PbPb) or 'pp'; R_asym 1.0-1.5."""
    out = {}
    for name in ("fig1_0", "fig1_1"):
        d = _load(hep, name)
        for dv in d["dependent_variables"]:
            q = _quals(dv)
            system = "pp" if "P P" in q.get("RE", "") else q["Centrality"].replace("%", "")
            out[(_pt_of(q), system)] = _series(d, dv)
    return out


def table3(hep):
    """Table 3 -> {(pt, gap): series}: PbPb 0-30 % - pp."""
    d = _load(hep, "fig2")
    return {(_pt_of(q), _gap_of(q)): _series(d, dv)
            for dv in d["dependent_variables"] for q in [_quals(dv)]}


def table4(hep):
    """Table 4 -> {(pt, gap): series over the centrality classes (x: 65, 40, 15 = 50-80,
    30-50, 0-30 %)}: PbPb - pp integrated over -2.4 < Delta eta < -0.6."""
    d = _load(hep, "fig3")
    return {(_pt_of(q), _gap_of(q)): _series(d, dv)
            for dv in d["dependent_variables"] for q in [_quals(dv)]}


def to_cms(values_02, norm="per-bin"):
    """The model in CMS's 0.2 bins (per unit Delta y, deta_asym.rebin) -> CMS's convention."""
    return np.asarray(values_02) * NORMS[norm]


def integral_cms(values_02, centres, norm="per-bin"):
    """Table 4's quantity from the model in CMS's 0.2 bins: the sum over the 10 bins centred
    -2.4 ... -0.6 in CMS's convention."""
    m = (centres > -2.45) & (centres < -0.55)
    return np.nansum(to_cms(values_02, norm)[..., m], axis=-1)


def chi2(model, model_err, data):
    """chi^2 and bins of the model against a series (model and data total errors in
    quadrature; bins with a finite model)."""
    ok = np.isfinite(model) & np.isfinite(model_err)
    e2 = data["total"][ok] ** 2 + model_err[ok] ** 2
    return float(np.sum((model[ok] - data["value"][ok]) ** 2 / e2)), int(ok.sum())
