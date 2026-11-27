#!/usr/bin/env python3
"""
example/prod_OO_isr/rehadronize_colorless.py

Hadronize the jets of a jet_frag file again with COLORLESS (JETSCAPE's string fragmentation without
colour flow), from the final partons the file stored (partons/: Matter/LBT's output with the recoils,
pstat 1, and the holes, pstat -1).  The companion of rehadronize_jets.py, which repeats iColored from
hadronization_input/.  With both, the same partons can be hadronized either way: the hadronization's
own effect on R_AA, at fixed energy loss.

    python rehadronize_colorless.py IN_hadrons_jet_frag.h5 --out OUT_hadrons_jet_frag.h5
    python rehadronize_colorless.py IN.h5 --out OUT.h5 --hadronize-xml ../analysis/OO-RHIC/refs/hadronize_pp19_icdecays.xml

Colorless (X-SCAPE ColorlessHadronization) takes the <JetHadronization> block of --hadronize-xml
(default ../config/hadronize.xml; eCMforHadronization from the source job's XML).  With take_recoil it
hadronizes the positive partons (pstat 0, 1; the O+O chain's status 1000, PythiaIsrGun's hard partons and
their Matter daughters, is relabelled 0 first, else Colorless drops them) and, separately, the holes into
hadrons with pstat -1
(stored with E > 0): an analysis subtracts those (analysis/OO-RHIC/auau_raa.py does).  The beam
remnants and the colour-balancing partons of iColored (hadronization_input/) are not used.

Output: a jet_frag file (jetscape.hadrons_h5) with one sample per event, the source's units/ event
record (pthat, window, weights, the hard partons' rapidities, Glauber values), initiators/ and partons/,
the source's cross sections and parton cut (attributes), and rehadronized_from / hadronization =
colorless.  Seeds: jetscape.hadronizer's per-unit scheme from --seed and the source's file name.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.join(os.path.dirname(HERE), "prod_AuAu_0_10")
_spec = importlib.util.spec_from_file_location("_run_prod", os.path.join(PROD, "run_prod.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)                 # also puts PyJetscape/python on sys.path

#: event-record columns carried over (when the source has them)
UNIT_KEYS = ("pthat", "pthat_bin", "event_weight", "parton_pt_lead", "parton_pt_sub", "parton_y_lead",
             "parton_y_sub", "glauber_b", "glauber_ncoll", "glauber_npart", "has_medium", "tau0_music",
             "sigma_gen", "sigma_err")
#: file attributes carried over
ATTR_KEYS = ("pthat_bins", "pthat_bin_sigma_gen", "pthat_bin_sigma_err", "pthat_bin_n_accepted",
             "pthat_bin_sigma_gen_raw", "pthat_bin_sigma_err_raw", "pthat_bin_acceptance", "pthat_bin_n_tried",
             "pthat_bin_n_kept", "pthat_bin_seeds", "parton_ymax", "parton_y_mode", "prod_user_xml",
             "prod_hard", "system", "eloss", "prod_seed", "subtract_hard_momentum", "hard_pair_string_veto")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jet_frag", help="a <stem>_hadrons_jet_frag.h5 with partons/")
    p.add_argument("--out", required=True, help="the new jet_frag file")
    p.add_argument("--hadronize-xml", default=os.path.join(os.path.dirname(HERE), "config", "hadronize.xml"),
                   dest="hadronize_xml", help="its <JetHadronization> (name colorless)")
    p.add_argument("--seed", type=int, default=1, help="base seed of the fragmentation (default 1)")
    p.add_argument("--n-frag", type=int, default=1, dest="n_frag",
                   help="fragmentations per event (oversamples of the same partons; default 1). Sample 0 has "
                        "the seed of a 1-sample run; analyses average the samples per event")
    p.add_argument("--build", default=rp.BUILD)
    p.add_argument("--main-xml", default=rp.MAIN_XML, dest="main_xml")
    p.add_argument("--workdir", default=None, help="scratch directory (default: next to --out)")
    p.add_argument("--force", action="store_true", help="overwrite --out")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    import h5py
    import hdf5plugin  # noqa: F401
    from jetscape import hadronizer as hz
    from jetscape import pyjetscape_core as core

    src, out = os.path.abspath(a.jet_frag), os.path.abspath(a.out)
    if os.path.exists(out) and not a.force:
        sys.exit(f"rehadronize_colorless.py: {out} exists (--force to overwrite)")
    with h5py.File(src) as h:
        if "partons/data" not in h:
            sys.exit(f"rehadronize_colorless.py: {src} has no partons/ (run with --store-partons)")
        n = int(h.attrs["nunits_written"])
        po = h["partons/offsets"][: n + 1]
        par = h["partons/data"][: po[-1]].astype(np.float64)
        io = h["initiators/offsets"][: n + 1] if "initiators/offsets" in h else None
        ini = h["initiators/data"][: io[-1]].astype(np.float64) if io is not None else None
        units = {k: h[f"units/{k}"][:n] for k in UNIT_KEYS if f"units/{k}" in h}
        attrs = {k: h.attrs[k] for k in ATTR_KEYS if k in h.attrs}
        ecm = hz.collision_ecm(str(h.attrs.get("prod_user_xml", "")))

    root = ET.parse(a.hadronize_xml).getroot()
    had = root.find("JetHadronization")
    if had is None or (had.findtext("name") or "").strip().lower() != "colorless":
        sys.exit(f"rehadronize_colorless.py: {a.hadronize_xml}: need <JetHadronization><name>colorless")
    if ecm:
        hz.set_hadronization_ecm(root, ecm)
    had_text = ET.tostring(had, encoding="unicode")

    workdir = os.path.abspath(a.workdir or (os.path.splitext(out)[0] + "_work"))
    os.makedirs(workdir, exist_ok=True)
    job_xml = os.path.join(workdir, "rehad_job.xml")
    with open(job_xml, "w") as f:
        f.write("<jetscape><nEvents>1</nEvents><debug>off</debug><remark>off</remark><vlevel>0</vlevel>"
                "<Random><seed>1</seed></Random>" + had_text + "</jetscape>")
    main_xml = os.path.join(workdir, "jetscape_main.xml")
    rp._absolute_parent_paths(a.main_xml, main_xml, a.build)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        jetscape = core.JetScapePerEvent()           # reads the XML only; Colorless runs on given partons
        jetscape.SetXMLMainFileName(main_xml)
        jetscape.SetXMLUserFileName(job_xml)
        jetscape.Init()
        colorless = core.create_module("ColorlessHadronization")
        colorless.Init()
        import uuid
        key = hz.uuid_key(str(uuid.uuid5(uuid.NAMESPACE_URL, os.path.basename(src))))   # reproducible per source
        h = hz.Hadronizer(core, colorless=colorless, base_seed=a.seed, prod=key)
        meta = {**attrs, "source": os.path.basename(src), "rehadronized_from": src, "hadronization": "colorless",
                "hadronize_xml": had_text, "base_seed": a.seed, "seed_scheme": "production_file",
                "hadrons_only": True, "legs": ["jet"], "music_input": "", "x_had": "none", "n_frag": a.n_frag,
                **hz.tag_attrs("jet_frag", core, {})}
        h.open("jet_frag", out, n_samples=a.n_frag, attrs=meta, keep_bits={"p": None, "x": None},
               eta_max=None, charged_only=False, positions=False, initiators=ini is not None, partons=True)
        n_neg = n_pos = 0
        try:
            for i in range(n):
                p_i = par[po[i]: po[i + 1]].copy()
                # PythiaIsrGun marks the hard partons 1000 and Matter's daughters keep it; Colorless takes
                # only 0, 1, 22, -1 and would drop them (README "Not working yet: Matter + LBT with
                # ColorlessHadronization", problem 1): hadronize them as shower partons
                p_i[p_i[:, 2] == 1000, 2] = 0
                ini_i = ini[io[i]: io[i + 1]] if ini is not None else None
                sc = {k: (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in units.items()}
                frags = h.fragments(i, p_i, [h.seed("jet_frag", i, k) for k in range(a.n_frag)],
                                    initiators=ini_i, **sc)
                for fr in frags:
                    ps = np.asarray(fr["pstat"]) if len(fr["pid"]) else np.zeros(0)
                    n_neg += int((ps == -1).sum()); n_pos += int((ps != -1).sum())
            h.set_attrs({"nevents": n, "complete_job": True})
        finally:
            h.close(complete=True)
    finally:
        os.chdir(cwd)
    print(f"rehadronize_colorless.py: {os.path.basename(src)} -> {out}: {n} events, "
          f"{n_pos / max(n * a.n_frag, 1):.1f} hadrons and {n_neg / max(n * a.n_frag, 1):.1f} hole hadrons (pstat -1) "
          f"per event and fragmentation, {a.n_frag} fragmentation(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
