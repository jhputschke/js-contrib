#!/usr/bin/env python3
"""
example/prod_OO_isr/rehadronize_icolored.py

Hadronize the jets of a jet_frag file again with our iCOLORED (X-SCAPE iColoredHadronization: Lund strings
along the stored colour tags), from the final partons the file stored (partons/), without beam remnants (or with the job's real ones,
--beam-ends stored):
iColored's stored-parton path (pyjetscape hadronize_partons -> HadronizePartons), which closes every open
colour line with <colour_closure> (fake: a soft end parton of 1 GeV along the beam, E = |p| = 1 GeV, not
sqrt(s)/6; reconnect: open ends joined within a shower by smallest invariant mass, then partners on the
holes' momenta, then joined across showers, then fakes).  The companion of rehadronize_colorless.py: the same
partons, here for files without hadronization_input/ (e.g. run_prod_jet_single.py, run_prod_vacuum.py).

    python rehadronize_icolored.py IN_hadrons_jet_frag.h5 --out OUT_hadrons_jet_frag.h5 --colour-closure fake

iColored's <JetHadronization> is the O+O production's (run_prod_oo_isr.py's job XML: decays on with
tau0 < 10 mm, colour_duplicates fail, reseed per event), with --colour-closure and --colour-duplicates.
Holes (pstat -1) are not hadronized (iColored drops them; their tags only steer the closure); the output keeps
partons/ for a parton-level hole subtraction.  The closure's partner copies (reconnect, thermal) are
partons on the holes' momenta with flipped colour, made inside iColored: not stored, but reconstructable
from partons/ (one per hole whose colour line is open).  Every parton is massless, E = |p| (as in the job).

Output: a jet_frag file (jetscape.hadrons_h5) with one sample per event, the source's units/ event record,
initiators/ and partons/, the cross sections and parton cut, rehadronized_from / hadronization = icolored,
and units/colour_* (the per-event colour audit).  Seeds: jetscape.hadronizer's per-unit scheme from --seed
and the source's file name (as rehadronize_colorless.py).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
             "sigma_gen", "sigma_err", "hydro_event", "hydro_ran", "reuse_index")
#: file attributes carried over
#: the O+O production's iColored block (run_prod_oo_isr.py job XMLs, e.g. B1 auau_med: jet_hadronization_xml),
#: without store_input; <colour_closure> and <colour_duplicates> from the command line
ICOLORED_XML = """<JetHadronization>
  <name>iColored</name>
  <take_recoil>1</take_recoil>
  <eCMforHadronization>200</eCMforHadronization>
  <reco_Mlevelmax>1</reco_Mlevelmax>
  <reco_Blevelmax>1</reco_Blevelmax>
  <reco_goldstone>0</reco_goldstone>
  <thermreco_distmax>5.0</thermreco_distmax>
  <shower_recofactor>1.0</shower_recofactor>
  <thermal_recofactor>1.0</thermal_recofactor>
  <recobias_t>0</recobias_t>
  <hydro_Tc>0.16</hydro_Tc>
  <eta_max_boost_inv>2.0</eta_max_boost_inv>
  <had_postprop>0.0</had_postprop>
  <part_prop>0.0</part_prop>
  <pythia_decays>on</pythia_decays>
  <tau0Max>10.0</tau0Max>
  <weak_decays>deprecated</weak_decays>
  <colour_duplicates>fail</colour_duplicates>
  <LinesToRead>
    Init:showProcesses = off
  </LinesToRead>
  <reco_hadrons_in_pythia>1</reco_hadrons_in_pythia>
  <additional_pythia_particles>0</additional_pythia_particles>
  <colour_closure>fake</colour_closure>
  <reseed_per_event>1</reseed_per_event>
</JetHadronization>"""
#: the per-event colour audit (jet_hadronization_colour_audit), stored as units/colour_<key>
AUDIT_KEYS = ("partons", "holes", "hole_energy", "unpaired_col", "unpaired_acol", "duplicate_tags",
              "relabelled_tags", "reconnected", "reconnected_cross", "partners", "partner_energy", "fakes",
              "fake_energy", "retries", "reseeded", "failed")
#: other seeds tried when Pythia cannot fragment an event
MAX_RESEED = 5
ATTR_KEYS = ("pthat_bins", "pthat_bin_sigma_gen", "pthat_bin_sigma_err", "pthat_bin_n_accepted",
             "pthat_bin_sigma_gen_raw", "pthat_bin_sigma_err_raw", "pthat_bin_acceptance", "pthat_bin_n_tried",
             "pthat_bin_n_kept", "pthat_bin_seeds", "parton_ymax", "parton_y_mode", "prod_user_xml",
             "prod_hard", "system", "eloss", "prod_seed", "subtract_hard_momentum", "hard_pair_string_veto",
             "reuse")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jet_frag", help="a <stem>_hadrons_jet_frag.h5 with partons/")
    p.add_argument("--out", required=True, help="the new jet_frag file")
    p.add_argument("--colour-closure", default="fake", dest="closure",
                   choices=("none", "fake", "reconnect", "reconnect_global", "thermal"),
                   help="iColored <colour_closure> (default fake)")
    p.add_argument("--colour-duplicates", default="fail", dest="duplicates", choices=("fail", "relabel"),
                   help="iColored <colour_duplicates> (default fail, as in production)")
    p.add_argument("--beam-ends", default="none", dest="beam_ends", choices=("none", "sqrts6", "stored"),
                   help="sqrts6: before iColored, end every open colour line of the kept partons (pstat >= 0) whose "
                        "tag is not on a hole with a remnant-like beam parton (dbar for an open colour, d for an open "
                        "anticolour; (0, 0, +-sqrt(s)/6), E = sqrt(s)/6, massless, sign from the open parton's pz; own "
                        "shower index), as Colorless's / Colored's fake remnants; the holes' lines are left to "
                        "--colour-closure.  Recorded in units/beam_ends, units/beam_end_energy, units/open_tags_before. "
                        "partons/ stores the original partons (without the beam ends).  stored: the job's real beam "
                        "remnants instead (files with hadronization_input/, run_prod_oo_isr.py), so iColored sees the "
                        "job's input before its closure and --colour-closure can be varied; see stored_remnants")
    p.add_argument("--seed", type=int, default=1, help="base seed of the fragmentation (default 1)")
    p.add_argument("--n-frag", type=int, default=1, dest="n_frag",
                   help="fragmentations per event (oversamples of the same partons; default 1). Sample 0 has the "
                        "seeds of a 1-sample run (identical result); sample j uses seed indices 100 j + retry. The "
                        "colour audit is sample 0's; analyses average the samples per event")
    p.add_argument("--build", default=rp.BUILD)
    p.add_argument("--main-xml", default=rp.MAIN_XML, dest="main_xml")
    p.add_argument("--workdir", default=None, help="scratch directory (default: next to --out)")
    p.add_argument("--force", action="store_true", help="overwrite --out")
    return p.parse_args(argv)


def add_beam_ends(p, ecm):
    """The partons p (N, 14) plus one beam parton per open colour line of the kept partons whose tag is not on a
    hole; returns (partons for iColored, number of ends, their energy, open tags before)."""
    keep = p[:, 2] >= 0
    col, acol = p[keep, 12].astype(np.int64), p[keep, 13].astype(np.int64)
    holes = p[p[:, 2] == -1]
    hole_tags = set(holes[:, 12].astype(np.int64)) | set(holes[:, 13].astype(np.int64))
    cs, acs = set(col[col != 0]), set(acol[acol != 0])
    open_col = [c for c in cs if c not in acs]
    open_acol = [t for t in acs if t not in cs]
    e = ecm / 6.0
    shower = (p[:, 0].max() + 1) if len(p) else 0
    rows = []
    kp = p[keep]
    for tag, is_col in [(c, True) for c in open_col] + [(t, False) for t in open_acol]:
        if tag in hole_tags:
            continue
        src = kp[(kp[:, 12] == tag) if is_col else (kp[:, 13] == tag)][0]
        sz = 1.0 if src[6] >= 0 else -1.0
        r = np.zeros(14)
        r[0], r[2], r[3], r[6] = shower, 0, e, sz * e
        if is_col:            # open colour c: an antiquark (dbar) carries acol c
            r[1], r[12], r[13] = -1, 0, tag
        else:                 # open anticolour a: a quark (d) carries col a
            r[1], r[12], r[13] = 1, tag, 0
        rows.append(r)
    out = np.vstack([p] + rows) if rows else p
    return out, len(rows), len(rows) * e, len(open_col) + len(open_acol)


def stored_remnants(p, inp):
    """The job's beam remnants from its stored Pythia input inp (hadronization_input/ of a run_prod_oo_isr.py
    file: what iColored handed Pythia AFTER its colour closure), as parton rows to add to the stored partons p
    (N, 14).  inp = the jet partons (pstat >= 0, matched by momentum; the closure may have retagged them), the
    closure's partner copies (on the holes' momenta), the remnants and the colour-balancing parton (the rest:
    massless, along the beam).  The partons as stored plus these rows are iColored's input before the closure:
    open colour lines are the holes' and the few the job's closure also had to close with partners or ends
    (returned: open tags not on a hole, informational).
    Returns (rows, number of remnants, their energy, open tags before the closure, open tags not on a hole)."""
    P = np.asarray(inp["p"], float).reshape(-1, 4)
    key = lambda x: tuple(np.round(x, 6))                     # noqa: E731
    kept = {key(r[4:7]) for r in p[p[:, 2] >= 0]}
    holes = p[p[:, 2] == -1]
    on_hole = {key(r[4:7]) for r in holes}
    # the closure's 1 GeV soft end partons (a job event that needed the fake fallback): E = |p| = 1 along the beam
    fake = (np.abs(P[:, 0] - 1.0) < 1e-9) & (np.hypot(P[:, 1], P[:, 2]) < 1e-9)
    rem = np.array([key(q[1:4]) not in kept and key(q[1:4]) not in on_hole for q in P], bool) & ~fake
    shower = (p[:, 0].max() + 1) if len(p) else 0
    rows = np.zeros((int(rem.sum()), 14))
    rows[:, 0], rows[:, 1], rows[:, 2] = shower, np.asarray(inp["pid"])[rem], 0
    rows[:, 3:7] = P[rem]
    rows[:, 12], rows[:, 13] = np.asarray(inp["col"])[rem], np.asarray(inp["acol"])[rem]
    allp = np.vstack([p[p[:, 2] >= 0], rows]) if len(rows) else p[p[:, 2] >= 0]
    from collections import Counter
    cc = Counter(int(c) for c in allp[:, 12] if c); ca = Counter(int(t) for t in allp[:, 13] if t)
    open_t = [c for c in cc if cc[c] > ca.get(c, 0)] + [t for t in ca if ca[t] > cc.get(t, 0)]
    hole_tags = set(holes[:, 12].astype(int)) | set(holes[:, 13].astype(int))
    return rows, len(rows), float(P[rem, 0].sum()), len(open_t), sum(t not in hole_tags for t in open_t)


def main(argv=None) -> int:
    a = parse_args(argv)
    import h5py
    import hdf5plugin  # noqa: F401
    from jetscape import hadronizer as hz
    from jetscape import pyjetscape_core as core

    src, out = os.path.abspath(a.jet_frag), os.path.abspath(a.out)
    if os.path.exists(out) and not a.force:
        sys.exit(f"rehadronize_icolored.py: {out} exists (--force to overwrite)")
    with h5py.File(src) as h:
        if "partons/data" not in h:
            sys.exit(f"rehadronize_icolored.py: {src} has no partons/ (run with --store-partons)")
        n = int(h.attrs["nunits_written"])
        po = h["partons/offsets"][: n + 1]
        par = h["partons/data"][: po[-1]].astype(np.float64)
        io = h["initiators/offsets"][: n + 1] if "initiators/offsets" in h else None
        ini = h["initiators/data"][: io[-1]].astype(np.float64) if io is not None else None
        units = {k: h[f"units/{k}"][:n] for k in UNIT_KEYS if f"units/{k}" in h}
        attrs = {k: h.attrs[k] for k in ATTR_KEYS if k in h.attrs}
        ecm = hz.collision_ecm(str(h.attrs.get("prod_user_xml", "")))
        unit_ids = h["units/unit"][:n] if "units/unit" in h else np.arange(n)
        jh_stored = str(h.attrs.get("jet_hadronization_xml", ""))
    hf_in = None
    if a.beam_ends == "stored":
        from jetscape.hadrons_h5 import HadronFile
        hf_in = HadronFile(src)
        if not hf_in.has_hadronization_input:
            sys.exit(f"rehadronize_icolored.py: --beam-ends stored needs hadronization_input/ in {src}")
        try:                                     # the job's own eCMforHadronization, as close to the job as possible
            ecm = float(ET.fromstring(jh_stored).findtext("eCMforHadronization").strip())
        except (ET.ParseError, AttributeError, ValueError):
            pass

    had = ET.fromstring(ICOLORED_XML)
    had.find("colour_closure").text = a.closure
    had.find("colour_duplicates").text = a.duplicates
    if ecm:
        had.find("eCMforHadronization").text = f"{ecm:g}"
    had_text = ET.tostring(had, encoding="unicode")

    workdir = os.path.abspath(a.workdir or (os.path.splitext(out)[0] + "_work"))
    os.makedirs(workdir, exist_ok=True)
    job_xml = os.path.join(workdir, "rehad_job.xml")
    with open(job_xml, "w") as f:
        # no automatic task list: iColored is only a module here (it would ask for <Hard><ISR>)
        f.write("<jetscape><enableAutomaticTaskListDetermination>false</enableAutomaticTaskListDetermination>"
                "<nEvents>1</nEvents><debug>off</debug><remark>off</remark><vlevel>0</vlevel>"
                "<Random><seed>1</seed></Random>" + had_text + "</jetscape>")
    main_xml = os.path.join(workdir, "jetscape_main.xml")
    rp._absolute_parent_paths(a.main_xml, main_xml, a.build)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        jetscape = core.JetScapePerEvent()           # reads the XML only; iColored runs on given partons
        jetscape.SetXMLMainFileName(main_xml)
        jetscape.SetXMLUserFileName(job_xml)
        jetscape.Init()
        icolored = core.create_module("iColoredHadronization")
        icolored.Init()
        import uuid
        key = hz.uuid_key(str(uuid.uuid5(uuid.NAMESPACE_URL, os.path.basename(src))))   # reproducible per source
        h = hz.Hadronizer(core, colorless=icolored, base_seed=a.seed, prod=key)   # hadronize_partons dispatches
        meta = {**attrs, "source": os.path.basename(src), "rehadronized_from": src, "hadronization": "icolored", "colour_closure": a.closure, "beam_ends": a.beam_ends,
                "hadronize_xml": had_text, "base_seed": a.seed, "seed_scheme": "production_file",
                "hadrons_only": True, "legs": ["jet"], "music_input": "", "x_had": "none", "n_frag": a.n_frag,
                **hz.tag_attrs("jet_frag", core, {})}
        h.open("jet_frag", out, n_samples=a.n_frag, attrs=meta, keep_bits={"p": None, "x": None},
               eta_max=None, charged_only=False, positions=False, initiators=ini is not None, partons=True)
        n_neg = n_pos = 0
        audits = []
        try:
            for i in range(n):
                p_i = par[po[i]: po[i + 1]].copy()
                p_i[p_i[:, 2] == 1000, 2] = 0     # PythiaIsrGun's label (iColored takes every pstat >= 0 anyway)
                ini_i = ini[io[i]: io[i + 1]] if ini is not None else None
                sc = {k: (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in units.items()}
                # Pythia can fail to fragment a configuration (e.g. a ministring below threshold); iColored
                # then throws (with closure fake it does not retry itself).  Retry with up to MAX_RESEED other
                # seeds (h.seed of sample 1, 2, ...), counted per event in units/colour_reseeded; duplicate
                # colour tags are not retried.
                # If every seed fails (a configuration Pythia cannot fragment, e.g. a massless charm quark on a
                # string with a collinear 1 GeV fake end: below the two-hadron threshold), the event is written
                # with no hadrons (its partons kept) and counted in units/colour_failed.
                if a.beam_ends == "sqrts6":
                    p_had, nb, eb, n_open = add_beam_ends(p_i, ecm or 200.0)
                    sc = {**sc, "beam_ends": nb, "beam_end_energy": eb, "open_tags_before": n_open}
                elif a.beam_ends == "stored":
                    rows, nb, eb, n_open, n_bad = stored_remnants(p_i, hf_in.hadronization_input(int(unit_ids[i])))
                    p_had = np.vstack([p_i, rows]) if len(rows) else p_i
                    sc = {**sc, "beam_ends": nb, "beam_end_energy": eb, "open_tags_before": n_open,
                          "open_tags_not_holes": n_bad}
                else:
                    p_had = p_i
                w = h.writers["jet_frag"]
                kw = {"partons": p_i} if getattr(w, "_par", None) is not None else {}   # partons/: the originals
                frags, seed, k0, failed, aud = [], None, 0, 0, {}
                for j in range(a.n_frag):
                    f_j, fail_j = None, 0
                    for k in range(MAX_RESEED + 1):
                        seed_k = int(h.seed("jet_frag", i, 100 * j + k))
                        try:
                            f_j = core.hadronize_partons(icolored, p_had, seed_k) if len(p_had) else hz._empty_hadrons()
                            break
                        except RuntimeError as err:
                            print(f"rehadronize_icolored.py: unit {i}, sample {j}, seed {k}: {err}", file=sys.stderr)
                            if "Pythia could not hadronize" not in str(err):
                                raise
                            if k == MAX_RESEED:
                                fail_j = 1
                                f_j = hz._empty_hadrons()
                    if j == 0:                     # sample 0: the seed, retries and colour audit of the event
                        seed, k0, failed = seed_k, k, fail_j
                        aud = {} if fail_j else dict(core.jet_hadronization_colour_audit(icolored))
                    frags.append(f_j)
                w.append_unit(frags, ini_i, **kw, unit=i, event=i, seed=seed, n_partons=len(p_i), **sc)
                aud["reseeded"] = k0
                aud["failed"] = failed
                audits.append([float(aud.get(k_, 0)) for k_ in AUDIT_KEYS])
                ps = np.asarray(frags[0]["pstat"]) if len(frags[0]["pid"]) else np.zeros(0)
                n_neg += int((ps == -1).sum()); n_pos += int((ps != -1).sum())
            h.set_attrs({"nevents": n, "complete_job": True})
        finally:
            h.close(complete=True)
    finally:
        os.chdir(cwd)
    A = np.array(audits).reshape(-1, len(AUDIT_KEYS))
    with h5py.File(out, "a") as f:                     # the colour audit per event, next to the event record
        for k, col in zip(AUDIT_KEYS, A.T):
            f.create_dataset(f"units/colour_{k}", data=col)
        f.attrs["colour_audit_mean"] = json.dumps({k: float(c.mean()) for k, c in zip(AUDIT_KEYS, A.T)})
    m = dict(zip(AUDIT_KEYS, A.mean(0)))
    print(f"rehadronize_icolored.py: {os.path.basename(src)} -> {out}: {n} events, "
          f"{n_pos / max(n, 1):.1f} hadrons per event; closure {a.closure}: open ends "
          f"{m['unpaired_col'] + m['unpaired_acol']:.1f}, reconnected {m['reconnected']:.1f} "
          f"(+{m['reconnected_cross']:.1f} across showers), partners {m['partners']:.1f} ({m['partner_energy']:.1f} GeV), "
          f"fakes {m['fakes']:.1f} ({m['fake_energy']:.1f} GeV), retries {m['retries']:.2f}, duplicate tags "
          f"{m['duplicate_tags']:.2f}, events reseeded {int((A[:, AUDIT_KEYS.index('reseeded')] > 0).sum())}, failed (no hadrons) {int(A[:, AUDIT_KEYS.index('failed')].sum())}, holes {m['holes']:.1f} ({m['hole_energy']:.1f} GeV, not hadronized) per event")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
