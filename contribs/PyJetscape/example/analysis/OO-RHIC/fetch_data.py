#!/usr/bin/env python3
"""
analysis/OO-RHIC/fetch_data.py -- download the HEPData tables of the tune and parse them.

    python fetch_data.py            # data/hepdata/*.yaml (raw) and data/data.json (parsed)
    python fetch_data.py --offline  # parse the stored YAML only

data.json: per observable the pT bin edges, values, statistical and systematic errors and
the units.  Observables:
    phenix_pi0        PHENIX pi0, E d3sigma/dp3 [mb GeV^-2], |eta| < 0.35   (fit)
    star_jets_eta0    STAR jets R = 0.5, d2sigma/dpT deta [pb/GeV], |eta| < 0.5   (fit)
    star_jets_eta1    the same, 0.5 < |eta| < 0.9   (fit)
    star_pi0          STAR pi0, E d3sigma/dp3 [mb GeV^-2], 0 < eta < 1   (validation)
    star_eta_pi0      STAR eta/pi0, 0 < eta < 1   (validation)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse

import yaml

import config as C

RAW = C.DATA_DIR / "hepdata"


def raw_path(rec, table):
    return RAW / f"{rec}_{table.replace(' ', '_').replace('(', '_').replace(')', '_')}.yaml"


def download(rec, table):
    """HEPData's table download (curl: HEPData refuses Python's urllib clients)."""
    import subprocess
    url = (f"https://www.hepdata.net/download/table/{rec}/"
           f"{urllib.parse.quote(table)}/yaml")
    text = subprocess.run(["curl", "-sfL", url], check=True, capture_output=True,
                          text=True, timeout=120).stdout
    raw_path(rec, table).write_text(text)
    return text


def _num(v):
    return float(str(v).replace(",", ""))


def _errs(entry):
    stat = sys_ = 0.0
    for e in entry.get("errors", []):
        lab = str(e.get("label", "")).lower()
        val = abs(_num(e["symerror"])) if "symerror" in e else \
            0.5 * (abs(_num(e["asymerror"]["plus"])) + abs(_num(e["asymerror"]["minus"])))
        if lab.startswith("stat"):
            stat = val
        else:
            sys_ = (sys_ ** 2 + val ** 2) ** 0.5
    return stat, sys_


def _bins(doc):
    for iv in doc["independent_variables"]:
        vals = iv["values"]
        if "low" in vals[0]:
            return [[_num(v["low"]), _num(v["high"])] for v in vals]
    raise ValueError("no binned independent variable")


def _series(dep):
    vals = [_num(v["value"]) for v in dep["values"]]
    errs = [_errs(v) for v in dep["values"]]
    return vals, [e[0] for e in errs], [e[1] for e in errs]


def parse(texts):
    out = {}
    d = yaml.safe_load(texts["phenix_pi0"])
    v, st, sy = _series(d["dependent_variables"][0])
    out["phenix_pi0"] = dict(edges=_bins(d), value=v, stat=st, sys=sy,
                             units="mb GeV^-2 c^3", eta=[-0.35, 0.35], kind="invariant",
                             norm=C.NORM_UNC["phenix_pi0"], label=r"PHENIX $\pi^0$, $|\eta|<0.35$",
                             ref="PRD 76, 051106 (2007); HEPData ins749394")
    d = yaml.safe_load(texts["star_jets"])
    edges = _bins(d)
    for k, dep in enumerate(d["dependent_variables"]):
        v, st, sy = _series(dep)
        eta = [0.0, 0.5] if k == 0 else [0.5, 0.9]
        out[f"star_jets_eta{k}"] = dict(
            edges=edges, value=v, stat=st, sys=sy, units="pb/GeV", abs_eta=eta,
            kind="jet_d2sigma", norm=C.NORM_UNC["star_jets"],
            label=(r"STAR jets $R=0.5$, " + (r"$|\eta|<0.5$" if k == 0 else r"$0.5<|\eta|<0.9$")),
            ref="arXiv:2603.28695; HEPData ins3137518")
    d = yaml.safe_load(texts["star_pi0"])
    v, st, sy = _series(d["dependent_variables"][0])
    out["star_pi0"] = dict(edges=_bins(d), value=v, stat=st, sys=sy, units="mb GeV^-2 c^3",
                           eta=[0.0, 1.0], kind="invariant", norm=None,
                           label=r"STAR $\pi^0$, $0<\eta<1$",
                           ref="PRC 81, 064904 (2010); HEPData ins840766")
    d = yaml.safe_load(texts["star_eta_pi0"])
    v, st, sy = _series(d["dependent_variables"][0])
    out["star_eta_pi0"] = dict(edges=_bins(d), value=v, stat=st, sys=sy, units="",
                               eta=[0.0, 1.0], kind="ratio", norm=None,
                               label=r"STAR $\eta/\pi^0$, $0<\eta<1$",
                               ref="PRC 81, 064904 (2010); HEPData ins840766")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--offline", action="store_true", help="parse the stored YAML only")
    a = p.parse_args(argv)
    RAW.mkdir(parents=True, exist_ok=True)
    texts = {}
    for key, (rec, table) in C.HEPDATA.items():
        path = raw_path(rec, table)
        if a.offline or path.exists():
            texts[key] = path.read_text()
        else:
            texts[key] = download(rec, table)
            print(f"downloaded {rec} '{table}' -> {path.name}")
    data = parse(texts)
    (C.DATA_DIR / "data.json").write_text(json.dumps(data, indent=1))
    for k, d in data.items():
        print(f"{k:16s} {len(d['value']):3d} bins, pT {d['edges'][0][0]}-{d['edges'][-1][1]} "
              f"GeV, {d['units']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
