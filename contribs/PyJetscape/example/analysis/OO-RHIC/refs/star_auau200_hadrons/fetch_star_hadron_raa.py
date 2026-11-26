#!/usr/bin/env python3
"""
refs/star_auau200_hadrons/fetch_star_hadron_raa.py -- STAR's high-pT hadron R_AA in central Au+Au at
200 GeV from HEPData, next to PHENIX's pi0 (refs/phenix_auau200_pi0/):

    python fetch_star_hadron_raa.py          # -> star_hadron_raa.json (this folder)

- pi+ + pi-, |y| < 0.5, 0-12 %: STAR, PRL 108, 072302 (2012), arXiv:1110.0579, Fig. 3 (HEPData ins930463,
  table Figure3.1), p_T 3-15 GeV/c; statistical and systematic errors separately; the p+p reference is
  STAR's measured p+p (sigma_NSD 30.0 +- 3.5 mb, i.e. a ~12 % normalisation not in the point errors).
- (h+ + h-)/2, |eta| < 0.5, 0-5 % and 10-20 %: STAR, PRL 91, 172302 (2003), nucl-ex/0305015, Fig. 3
  (HEPData ins619063, Figure3withBins), p_T up to 10 GeV/c in the table; one error (stat + sys); the N_bin
  normalisation (0-5 %: +-16 %) is not in the point errors. Relative to STAR's measured p+p spectrum.
(HEPData's metadata for ins619063 says Pb+Pb 2.76 TeV; the table and the paper are Au+Au 200 GeV.)
Also the measured p+p spectra these R_AA use (pp_pion, pp_hadron: NSD invariant yields), for the
p+p-reference check (pp_ref_check.py).  The downloaded YAML stays local (git-ignored); the JSON is
committed.
"""
import io
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
SUB = "https://www.hepdata.net/download/submission/ins{}/1/yaml"


def table(ins, name):
    f = HERE / f"ins{ins}_{name}.yaml"
    if not f.exists():
        req = urllib.request.Request(SUB.format(ins), headers={"User-Agent": "curl/8.5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, tarfile.open(fileobj=io.BytesIO(r.read())) as t:
            m = next(x for x in t.getmembers() if x.name.endswith(f"/{name}.yaml"))
            f.write_bytes(t.extractfile(m).read())
    return yaml.safe_load(f.read_text())


def points(d, k):
    xs = d["independent_variables"][0]["values"]
    out = []
    for x, v in zip(xs, d["dependent_variables"][k]["values"]):
        e = {q["label"]: q["symerror"] for q in v["errors"]}
        out.append({"lo": x["low"], "hi": x["high"], "value": v["value"], **{
            {"Stat error": "stat", "Sys error": "syst", "sys + stat": "stat_syst"}[lab]: val for lab, val in e.items()}})
    return out


def main():
    pi = table(930463, "Figure3.1")
    h = table(619063, "Figure3withBins")
    cls = [dv["qualifiers"][0]["value"].split()[0] for dv in h["dependent_variables"]]
    res = {"pion": {"source": "STAR, PRL 108, 072302 (2012), arXiv:1110.0579, Fig. 3 (HEPData ins930463 Figure3.1)",
                    "observable": "R_AA of pi+ + pi-, |y| < 0.5, Au+Au 200 GeV 0-12 %, measured p+p reference",
                    "normalisation": "p+p sigma_NSD 30.0 +- 3.5 mb, not in the point errors",
                    "points": points(pi, 0)},
           "hadron": {"source": "STAR, PRL 91, 172302 (2003), nucl-ex/0305015, Fig. 3 (HEPData ins619063 Figure3withBins)",
                      "observable": "R_AA of (h+ + h-)/2, |eta| < 0.5, Au+Au 200 GeV, measured p+p reference",
                      "normalisation": "N_bin: 0-5 % +-16 %, 10-20 % +-17 %, not in the point errors",
                      "centrality": {c: points(h, k) for k, c in enumerate(cls) if c in ("0-5%", "10-20%")}}}
    # the measured p+p spectra these R_AA use (NSD invariant yields; x sigma_NSD 30 mb -> mb GeV^-2)
    pp = table(930463, "Figure1.1")
    xs = pp["independent_variables"][0]["values"]
    dv = {d_["qualifiers"][0]["value"].split(" ")[0]: d_["values"] for d_ in pp["dependent_variables"]}
    pip, pim = dv["$\\pi^+$"], dv["$\\pi^-$"]
    def e_(v, lab):
        return next(q["symerror"] for q in v["errors"] if q["label"] == lab)
    res["pp_pion"] = {"source": "STAR 1110.0579 Fig. 1 (HEPData ins930463 Figure1.1): (pi+ + pi-)/2, |y| < 0.5, NSD p+p",
                      "units": "1/(2 pi pT) d2N/dpT dy [GeV^-2], NSD; sigma_NSD 30.0 +- 3.5 mb",
                      "points": [{"lo": x["low"], "hi": x["high"], "value": 0.5 * (a_["value"] + b_["value"]),
                                  "stat": 0.5 * float(np.hypot(e_(a_, "Stat error"), e_(b_, "Stat error"))),
                                  "syst": 0.5 * (e_(a_, "Sys error") + e_(b_, "Sys error"))}
                                 for x, a_, b_ in zip(xs, pip, pim)]}
    f1 = table(619063, "Figure1")
    hp = next(d_ for d_ in f1["dependent_variables"] if d_["qualifiers"][0]["value"].startswith("p+p (NSD)"))
    hx = f1["independent_variables"][0]["values"]
    res["pp_hadron"] = {"source": "STAR nucl-ex/0305015 Fig. 1 (HEPData ins619063 Figure1): (h+ + h-)/2, |eta| < 0.5, NSD p+p",
                        "units": "1/(2 pi pT) d2N/dpT deta [GeV^-2], NSD; sigma_NSD 30.0 +- 3.5 mb",
                        "points": [{"lo": x["low"], "hi": x["high"], "value": v["value"], "stat_syst": v["errors"][0]["symerror"]}
                                   for x, v in zip(hx, hp["values"]) if v["value"] != "--"]}
    (HERE / "star_hadron_raa.json").write_text(json.dumps(res, indent=1))
    for q in res["pion"]["points"]:
        if q["lo"] >= 5:
            print(f"pi  {q['lo']:5.2f}-{q['hi']:5.2f}: {q['value']:.3f} +- {q['stat']:.3f} +- {q['syst']:.3f}")
    for c, ps in res["hadron"]["centrality"].items():
        print(f"h {c}: " + " ".join(f"{q['lo']:g}-{q['hi']:g}:{q['value']:.2f}" for q in ps if q["lo"] >= 4.4))
    print("->", HERE / "star_hadron_raa.json")


if __name__ == "__main__":
    main()
