#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/run_prod_vacuum.py

The vacuum (pp) reference of run_prod_jet.py: the same PythiaGun -- the <PythiaGun> block of
the job's user XML with run_prod_jet.py's pTHat windows, parton rapidity cut and trigger
options -- and Matter's shower in vacuum (<in_vac> 1, recoils off; the user XML's other
<Eloss> and <Matter> settings), with no initial state, hydro, LBT or liquefier.  Colorless
hadronizes the final partons --n-frag times per event with run_prod_jet.py --hadrons-only's
code and seed scheme (jetscape.hadronizer) and the <JetHadronization> block of the shared
../config/hadronize.xml, its eCMforHadronization set to the PythiaGun's eCM as in every
production (jetscape.hadronizer.set_hadronization_ecm).  The hard vertex is the origin.

One call = one job = one seed:

    <stem>_hadrons_jet_frag.h5   the hadrons of every fragmentation; units/ the event record
                                 (pthat, window, weight, the leading partons, with trigger Z
                                 the Z and its leptons: jetscape.hadrons_h5.Z_COLUMNS),
                                 initiators/, partons/; attributes: the windows' cross
                                 sections, the trigger, vacuum = True.  HadronFileReader and
                                 hadronize.py read it as a --hadrons-only jet file.
    <stem>.json, <stem>.xml      the summary and the job XML that ran

    conda activate js_fno
    python run_prod_vacuum.py --events 1000 --seed 1 --user-xml ../config/PbPb_0_30_ZJet.xml \\
        --pthat-bins 20-30,30-35,35-45,45-55,55-70,70-200 \\
        --lepton-ptmin 20 --lepton-etamax 2.5 --n-frag 10
                                     # the pp reference of the Pb+Pb Z+jet production
    python run_prod_vacuum.py --events 100 --seed 1 --pthat-bins 10-20,20-30,30-40
                                     # QCD jets, the Au+Au production's PythiaGun and Matter
    python run_prod_vacuum.py --events 1 --seed 1 --dry-run     # check, print the plan

Event i uses pTHat window i mod K, so --events must be a multiple of the windows.  Use the
seeds of the medium production's campaign only for paired comparisons (the same Pythia
events); for more statistics give the reference its own seeds.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import platform
import socket
import sys
import time
import uuid as uuidlib
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))

# run_prod_jet.py's PythiaGun settings, trigger, event record and cross sections; loading it
# also loads ../prod_AuAu_0_10/run_prod.py (seeds, working directory) and puts
# PyJetscape/python on sys.path.
_spec = importlib.util.spec_from_file_location("_run_prod_jet",
                                               os.path.join(HERE, "run_prod_jet.py"))
rj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rj)
rp = rj.rp

#: <Eloss> settings taken over from the user XML (its modules are not: Matter only)
ELOSS_KEEP = ("deltaT", "maxT", "tStart", "mutex")
#: general settings taken over from the user XML
GENERAL_KEEP = ("debug", "remark", "vlevel", "nEvents_printout",
                "enableAutomaticTaskListDetermination")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", type=int, default=10, help="events in this job (default 10)")
    p.add_argument("--outdir", default=os.path.join(HERE, "out_vacuum"),
                   help="output directory (default: ./out_vacuum next to this script)")
    p.add_argument("--out", default=None,
                   help="output stem name (default: vacuum_jet[_<campaign>]_seed<NNNN>.h5, "
                        "or vacuum_jet_<campaign>_<NNNN>.h5 with --seed 0)")
    p.add_argument("--build", default=os.path.join(rp.XSCAPE, "build_gpu"),
                   help="X-SCAPE build tree (default: build_gpu)")
    p.add_argument("--main-xml", default=os.path.join(rp.XSCAPE, "config", "jetscape_main.xml"),
                   dest="main_xml")
    p.add_argument("--user-xml", default=rj.USER_XML, dest="user_xml",
                   help="the medium production's user XML, whose <Hard><PythiaGun> and "
                        "<Eloss> (Matter) this job takes (default: run_prod_jet.py's)")
    p.add_argument("--pthat-min", type=float, default=None, dest="pthat_min",
                   help="PythiaGun pTHatMin [GeV] (default: the XML's)")
    p.add_argument("--pthat-max", type=float, default=None, dest="pthat_max",
                   help="PythiaGun pTHatMax [GeV] (default: the XML's)")
    p.add_argument("--pthat-bins", default=None, dest="pthat_bins",
                   help="PythiaGun pTHat windows, e.g. 40-60,60-80 [GeV]: event i uses window "
                        "i mod K (as run_prod_jet.py)")
    p.add_argument("--parton-ymax", type=float, default=None, dest="parton_ymax",
                   help="as run_prod_jet.py: the handed-over partons' rapidity cut")
    p.add_argument("--parton-y-mode", choices=("leading", "both", "any"), default="leading",
                   dest="parton_y_mode")
    p.add_argument("--trigger", choices=("none", "Z"), default=None,
                   help="as run_prod_jet.py (default: the XML's <trigger>)")
    p.add_argument("--trigger-ymax", type=float, default=None, dest="trigger_ymax",
                   metavar="Y", help="trigger Z: |y_Z| < Y")
    p.add_argument("--lepton-ptmin", type=float, default=None, dest="lepton_ptmin",
                   metavar="PT", help="trigger Z: both leptons with pT > PT [GeV]")
    p.add_argument("--lepton-etamax", type=float, default=None, dest="lepton_etamax",
                   metavar="ETA", help="trigger Z: both leptons with |eta| < ETA")
    h = p.add_argument_group("hadrons", "Colorless on the final partons, as run_prod_jet.py "
                             "--hadrons-only does for the jet")
    h.add_argument("--hadronize-xml", default=rj.HADRONIZE_XML,
                   dest="hadronize_xml",
                   help="its <JetHadronization> block (eCMforHadronization becomes the "
                        "PythiaGun's eCM); default: ../config/hadronize.xml")
    h.add_argument("--hadronize-seed", type=int, default=1, dest="hadronize_seed",
                   help="base seed of the fragmentation (default 1); every event's seeds "
                        "derive from it and this job's file uuid")
    h.add_argument("--n-frag", type=int, default=1, dest="n_frag",
                   help="Colorless fragmentations per event (default 1)")
    h.add_argument("--hadron-keep-bits-p", type=int, default=None, dest="hadron_keep_bits_p",
                   help="mantissa bits kept for the momenta (hadronize.py --keep-bits-p)")
    h.add_argument("--hadron-keep-bits-x", type=int, default=None, dest="hadron_keep_bits_x",
                   help="mantissa bits kept for the positions (hadronize.py --keep-bits-x)")
    h.add_argument("--eta-max", type=float, default=None, dest="eta_max",
                   help="store only particles with |eta| < ETA_MAX")
    h.add_argument("--charged", action="store_true",
                   help="store only charged particles (jetscape.hadrons_h5.CHARGED)")
    h.add_argument("--no-x", action="store_true", dest="no_x", help="store no positions")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="check the XMLs, write the job XML, print the plan; do not run")
    rp.add_seed_args(p, "Pythia, Matter")
    rp.add_workdir_args(p)
    return p.parse_args()


def check_args(a) -> None:
    """The options against each other (sys.exit), with run_prod_jet.py's checks."""
    a.hard, a.reuse, a.jets_per_bin = "pythia", 1, 1
    problems = rp.seed_args_problems(a)
    if a.events < 1:
        problems.append("--events must be >= 1")
    if a.n_frag < 1:
        problems.append("--n-frag must be >= 1")
    for k in ("hadron_keep_bits_p", "hadron_keep_bits_x"):
        v = getattr(a, k)
        if v is not None and not 1 <= v <= 23:
            problems.append(f"--{k.replace('_', '-')} must be 1..23")
    if a.eta_max is not None and not a.eta_max > 0:
        problems.append("--eta-max must be > 0")
    try:
        if a.pthat_bins is not None:
            k = len(rj.parse_pthat_bins(a.pthat_bins))
            if a.events % k:
                raise ValueError(f"--events {a.events} is not a multiple of the {k} pTHat "
                                 f"windows (event i uses window i mod {k})")
        rj.check_pthat_bins(a)
        rj.check_parton_ycut(a)
        rj.check_trigger(a)
    except ValueError as exc:
        problems.append(str(exc))
    pythia = rp.pythia_data_problem()
    if pythia:
        problems.append(pythia)
    if problems:
        sys.exit("run_prod_vacuum.py: cannot start:\n  " + "\n  ".join(problems))


def job_xml(a, stem: str):
    """The vacuum job XML from the user XML: its general settings, <PythiaGun> (with this
    job's windows, cuts and trigger) and <Eloss> with Matter alone, in vacuum -> (path, eCM,
    Matter's settings as text)."""
    src = ET.parse(a.user_xml).getroot()
    pg, eloss = src.find("Hard/PythiaGun"), src.find("Eloss")
    matter = None if eloss is None else eloss.find("Matter")
    if pg is None or matter is None:
        sys.exit(f"run_prod_vacuum.py: {a.user_xml} needs <Hard><PythiaGun> and "
                 "<Eloss><Matter>")
    root = ET.Element("jetscape")
    for tag in GENERAL_KEEP:
        if src.find(tag) is not None:
            root.append(copy.deepcopy(src.find(tag)))
    for node in src:
        if node.tag.startswith("JetScapeWriter"):    # the template's writer switches (off)
            root.append(copy.deepcopy(node))
    rj._set(root, "nEvents", a.events)
    rj._set(root, "setReuseHydro", "false")
    rj._set(root, "nReuseHydro", 1)
    ET.SubElement(ET.SubElement(root, "Random"), "seed").text = str(a.seed)

    pg = copy.deepcopy(pg)
    ET.SubElement(root, "Hard").append(pg)
    problems = rj.configure_pythia_gun(a, pg)
    if problems:
        sys.exit(f"run_prod_vacuum.py: {a.user_xml}:\n  " + "\n  ".join(problems))
    ecm = float(rj._text(pg, "eCM"))

    el = ET.SubElement(root, "Eloss")
    for tag in ELOSS_KEEP:
        if eloss.find(tag) is not None:
            el.append(copy.deepcopy(eloss.find(tag)))
    matter = copy.deepcopy(matter)
    rj._set(matter, "in_vac", 1)
    rj._set(matter, "recoil_on", 0)
    el.append(matter)

    path = stem + ".xml"
    ET.ElementTree(root).write(path)
    settings = ", ".join(f"{n.tag} {n.text.strip()}" for n in matter
                         if n.text and n.text.strip() and n.tag != "name")
    return path, ecm, settings


def hadronize_blocks(a, ecm: float):
    """--hadronize-xml's <JetHadronization> with eCMforHadronization = ``ecm`` -> (the block
    as recorded, the <jetscape> document the module reads, the file's own value)."""
    from jetscape import hadronizer as hz

    root = ET.parse(a.hadronize_xml).getroot()
    had = root.find("JetHadronization")
    if had is None:
        sys.exit(f"run_prod_vacuum.py: {a.hadronize_xml} has no <JetHadronization>")
    if (rj._text(had, "name") or "").lower() != "colorless":
        sys.exit(f"run_prod_vacuum.py: {a.hadronize_xml}: <JetHadronization><name> must be "
                 "colorless")
    was = hz.set_hadronization_ecm(root, ecm)
    text = ET.tostring(had, encoding="unicode")
    return text, "<jetscape>" + text + "</jetscape>", was


def main() -> int:
    a = parse_args()
    for k in ("build", "outdir", "main_xml", "user_xml", "hadronize_xml"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    check_args(a)
    os.makedirs(a.outdir, exist_ok=True)
    out_h5 = os.path.join(a.outdir, rp.output_name(a, "vacuum_jet"))
    stem = os.path.splitext(out_h5)[0]
    rp.resolve_seed(a, out_h5, record=not a.dry_run)
    xml, ecm, matter_desc = job_xml(a, stem)
    had_text, had_doc, had_was = hadronize_blocks(a, ecm)

    if a.pthat_windows:
        hard_desc = ("PythiaGun pTHat windows "
                     + ", ".join(f"{lo:g}-{hi:g}" for lo, hi in a.pthat_windows) + " GeV")
    else:
        root = ET.parse(xml).getroot()
        hard_desc = (f"PythiaGun pTHat {rj._text(root, 'Hard/PythiaGun/pTHatMin')}-"
                     f"{rj._text(root, 'Hard/PythiaGun/pTHatMax')} GeV")
    if a.parton_ymax is not None:
        hard_desc += f", partons |y| < {a.parton_ymax:g} ({a.parton_y_mode})"
    if a.trigger_kind != "none":
        hard_desc += f", trigger {rj.trigger_desc(a)}"
    print(f"prod_vacuum: {a.events} event(s), seed {a.seed}"
          + (" (from OS entropy)" if a.seed_source == "os_entropy" else "")
          + (f", campaign {a.campaign} #{a.index}" if a.campaign else "")
          + f", p+p {ecm:g} GeV, {hard_desc}, Matter in vacuum, {a.n_frag} "
          f"fragmentation(s) per event")
    print(f"  build    {a.build}")
    print(f"  job XML  {xml}")
    print(f"  Matter   {matter_desc}")
    print(f"  hadrons  Colorless from {a.hadronize_xml}, eCMforHadronization {ecm:g}"
          + (f" (the file has {had_was})" if had_was and float(had_was) != ecm else ""))
    print(f"  output   {stem}_hadrons_jet_frag.h5")
    if a.dry_run:
        return 0

    import numpy as np
    import jetscape as js
    from jetscape import hadronizer as hz
    from jetscape import pyjetscape_core as core
    from jetscape.hadrons_h5 import Z_COLUMNS
    from jetscape.particlize_h5 import FINAL_PARTON_COLUMNS
    from jetscape.showers import initiators_from_manager

    if a.trigger_kind == "Z" and not getattr(core, "PYTHIA_GUN_HAS_Z_TRIGGER", False):
        sys.exit("run_prod_vacuum.py: trigger Z needs X-SCAPE's PythiaGun with <trigger> "
                 "(branch zjet) and PyJetscape built against it")
    file_uuid = str(uuidlib.uuid4())
    provenance = {
        "prod": "PyJetscape/example/prod_AuAu_0_10_jet/run_prod_vacuum.py",
        **rp.seed_provenance(a),
        "prod_host": socket.gethostname(),
        "prod_platform": platform.platform(),
        "prod_build": a.build,
        "prod_user_xml": open(xml).read(),
        "prod_template_xml": a.user_xml,
        "prod_hard": hard_desc,
        "system": f"p+p {ecm:g} GeV (vacuum reference)",
        "hard_vertex": "origin (no initial state)",
        "jet": f"Matter in vacuum (in_vac 1, recoil_on 0): {matter_desc}",
        "vacuum": True,
        **({"pthat_bins": np.asarray(a.pthat_windows, dtype=np.float64)}
           if a.pthat_windows else {}),
        **({"parton_ymax": a.parton_ymax, "parton_y_mode": a.parton_y_mode}
           if a.parton_ymax is not None else {}),
        **({"trigger": a.trigger_kind,
            **{k: getattr(a, k) for k in rj.TRIGGER_CUTS if getattr(a, k) is not None}}
           if a.trigger_kind != "none" else {}),
        **({"min_partons": int(a.min_partons)} if a.min_partons is not None else {}),
    }

    workdir = rp.job_workdir(a, out_h5)
    main_xml = rp.enter_workdir(a, workdir, xml)
    print(f"  workdir  {workdir}")
    jetscape = js.JetScapePerEvent()
    jetscape.SetXMLMainFileName(main_xml)
    jetscape.SetXMLUserFileName(xml)
    jetscape.Init()
    tasks = {t.GetId(): t for t in jetscape.GetTaskList()}
    gun = rj.open_pthat_bins(a, jetscape) or tasks.get("PythiaGun")
    if gun is None:
        sys.exit(f"run_prod_vacuum.py: no PythiaGun task (tasks: {sorted(tasks)})")
    mgr = core.JetScapeSignalManager.Instance().GetJetEnergyLossManagerPointer()
    if mgr is None:
        sys.exit("run_prod_vacuum.py: no JetEnergyLossManager (no <Eloss> task?)")

    # Colorless outside the task list, as run_prod_jet.py --hadrons-only: the task counter is
    # set back, so the per-event Matter clones get the IDs (and seeds) they would have had.
    n_task = core.next_task_number()
    core.xml_add_user_blocks(had_doc)
    colorless = core.create_module("ColorlessHadronization")
    colorless.Init()
    core.set_next_task_number(n_task)
    h = hz.Hadronizer(core, colorless=colorless, base_seed=a.hadronize_seed,
                      prod=hz.uuid_key(file_uuid))
    out_frag = f"{stem}_hadrons_jet_frag.h5"
    attrs = {"source": os.path.basename(stem), "source_uuid": file_uuid,
             "hadronize_xml": had_text, "base_seed": a.hadronize_seed,
             "stored_seeds": False, "seed_scheme": "production_file", "hadrons_only": True,
             "legs": ["jet"], "music_input": "", "x_had": "none", **provenance,
             **hz.tag_attrs("jet_frag", core, {})}
    h.open("jet_frag", out_frag, n_samples=a.n_frag, attrs=attrs,
           keep_bits={"p": a.hadron_keep_bits_p, "x": a.hadron_keep_bits_x},
           eta_max=a.eta_max, charged_only=a.charged, positions=not a.no_x,
           initiators=True, partons=True)

    t_job = time.time()
    n = 0
    n_had = 0.0
    finished = False
    windows = None
    try:
        jetscape.ExecInit()
        for i in range(jetscape.GetNumberOfEvents()):
            t0 = time.time()
            jetscape.ExecPerEvent()
            d = rj.pthat_diag(gun, a.trigger_kind)
            partons = np.asarray(mgr.final_partons_numpy(), dtype=np.float64).reshape(
                -1, len(FINAL_PARTON_COLUMNS))
            ini = initiators_from_manager(mgr)
            d["n_showers"] = len(ini)
            frags = h.fragments(i, partons, [h.seed("jet_frag", i, s) for s in range(a.n_frag)],
                                initiators=ini, **d)
            jetscape.ClearPerEvent()
            n += 1
            per_frag = sum(len(f["pid"]) for f in frags) / a.n_frag
            n_had += per_frag
            msg = (f"prod_vacuum: event {i + 1}/{a.events} done in {time.time() - t0:.2f} s, "
                   f"{len(ini)} showers, {len(partons)} final partons, {per_frag:.1f} "
                   f"particles/fragmentation, pTHat {d['pthat']:.1f} GeV (window "
                   f"{d['pthat_bin']})")
            if "z_pt" in d:
                msg += f", Z pT {d['z_pt']:.1f} GeV y {d['z_y']:+.2f}"
            print(msg, flush=True)
        windows = rj.pthat_windows_final(gun)
        jetscape.Finish()
        finished = True
    finally:
        h.set_attrs({**(windows or {}), "prod_wall_s_total": time.time() - t_job,
                     "nevents": n, "complete_job": finished})
        h.close(complete=finished and n == a.events)

    summary = {"out": out_frag, "seed": a.seed, "seed_source": a.seed_source,
               "campaign": a.campaign, "index": a.index, "system": provenance["system"],
               "hard": hard_desc, "events_requested": a.events, "events_written": n,
               "n_frag": a.n_frag, "particles_per_fragmentation": round(n_had / max(n, 1), 1),
               "hadronize_base_seed": a.hadronize_seed, "file_uuid": file_uuid,
               "wall_s": round(time.time() - t_job, 1)}
    if a.trigger_kind != "none":
        summary.update(trigger=a.trigger_kind, z_columns=list(Z_COLUMNS),
                       **{k: getattr(a, k) for k in rj.TRIGGER_CUTS
                          if getattr(a, k) is not None})
    if windows is not None:
        summary.update(pthat_bins=windows["pthat_bins"].tolist(),
                       pthat_bin_sigma_gen_mb=list(windows["pthat_bin_sigma_gen"]),
                       pthat_bin_seeds=[int(x) for x in windows["pthat_bin_seeds"]])
        if "pthat_bin_acceptance" in windows:
            summary.update(pthat_bin_acceptance=[round(float(x), 4) for x in
                                                 windows["pthat_bin_acceptance"]],
                           pthat_bin_n_tried=[int(x) for x in windows["pthat_bin_n_tried"]])
    with open(stem + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print("prod_vacuum:", json.dumps(summary))
    rp.leave_workdir(a, workdir, n == a.events)
    return 0 if n == a.events else 1


if __name__ == "__main__":
    raise SystemExit(main())
