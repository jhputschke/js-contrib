#!/usr/bin/env python3
"""
example/prod_AuAu_0_10_jet/run_prod_jet_single.py

Jets in a medium with ONE hydro leg: the initial state and MUSIC of the user XML (the first
<Hydro><MUSIC> block, the background leg of run_prod_jet.py), the jet's energy loss in it (the
user XML's <Eloss>: Matter + LBT, recoils and holes), no liquefier, no second (jet) hydro leg, and
Colorless on the final partons in the job.  For jet and hadron suppression without the medium's
hydrodynamic response to the jet (JETSCAPE's recoil + hole picture, e.g. arXiv:2204.01163): the
medium does not depend on the jet, so one hydro serves --reuse jets (setReuseHydro), which makes
the job a few times cheaper than run_prod_jet.py's two legs.

The hard process is the user XML's <Hard><PythiaGun> with run_prod_jet.py's pTHat windows, parton
rapidity cut and trigger options (its LinesToRead set Pythia's ISR/MPI); X-SCAPE places the hard
vertex at a binary collision of the initial state.  Colorless hadronizes the final partons --n-frag
times per event with run_prod_vacuum.py's code (jetscape.hadronizer) and the <JetHadronization>
block of --hadronize-xml (eCMforHadronization set to the PythiaGun's eCM).

One call = one job = one seed:

    <stem>_hadrons_jet_frag.h5   the jet's hadrons of every fragmentation (recoils included);
                                 units/ the event record (pthat, window, weight, the leading
                                 partons), initiators/, partons/ (with the holes, pstat -1, for the
                                 subtraction); attributes: the windows' cross sections, reuse
    <stem>.json, <stem>.xml      the summary and the job XML that ran

    python run_prod_jet_single.py --events 20 --seed 1 --reuse 10 --pthat-bins 10-20,20-30 \\
        --user-xml ../config/AuAu_MCGlauber_MUSIC_0_10_jet.xml
    python run_prod_jet_single.py --events 1 --seed 1 --dry-run         # check, print the plan

--xml-set PATH=VALUE changes an element of the user XML first (e.g.
Eloss/Matter/QhatParametrizationType=5).  Event i uses pTHat window i mod K and hydro event
i // reuse, so --events must be a multiple of the windows and of --reuse.  No bulk hadrons (no
iSS): use run_prod_jet.py for those.
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

# run_prod_vacuum.py (its hadronization blocks) and through it run_prod_jet.py (PythiaGun
# settings, trigger, event record, cross sections) and ../prod_AuAu_0_10/run_prod.py (seeds,
# working directory)
_spec = importlib.util.spec_from_file_location("_run_prod_vacuum",
                                               os.path.join(HERE, "run_prod_vacuum.py"))
rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rv)
rj, rp = rv.rj, rv.rp

#: user-XML blocks dropped: the second leg, the liquefier, in-job hadronization (done here)
DROP = ("Liquefier", "JetHadronization", "SoftParticlization")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", type=int, default=10, help="events (jets) in this job (default 10)")
    p.add_argument("--reuse", type=int, default=1,
                   help="jets per hydro event (setReuseHydro); 1 = a new medium every event")
    p.add_argument("--outdir", default=os.path.join(HERE, "out_single"),
                   help="output directory (default: ./out_single next to this script)")
    p.add_argument("--out", default=None,
                   help="output stem name (default: single_jet[_<campaign>]_seed<NNNN>.h5)")
    p.add_argument("--build", default=rp.BUILD,
                   help="X-SCAPE build tree (default: $JETSCAPE_BUILD, else build_gpu)")
    p.add_argument("--main-xml", default=rp.MAIN_XML, dest="main_xml")
    p.add_argument("--user-xml", default=rj.USER_XML, dest="user_xml",
                   help="user XML with <IS>, <Hard><PythiaGun>, <Hydro><MUSIC> (the first one is "
                        "used) and <Eloss> (default: run_prod_jet.py's)")
    p.add_argument("--xml-set", action="append", default=[], dest="xml_set", metavar="PATH=VALUE",
                   help="set an element of the user XML (path below <jetscape>, e.g. "
                        "IS/MCGlauber/b_max=4.7); repeatable")
    p.add_argument("--pthat-min", type=float, default=None, dest="pthat_min")
    p.add_argument("--pthat-max", type=float, default=None, dest="pthat_max")
    p.add_argument("--pthat-bins", default=None, dest="pthat_bins",
                   help="PythiaGun pTHat windows, e.g. 10-20,20-30 [GeV]: event i uses window i mod K")
    p.add_argument("--parton-ymax", type=float, default=None, dest="parton_ymax")
    p.add_argument("--parton-y-mode", choices=("leading", "both", "any"), default="leading",
                   dest="parton_y_mode")
    p.add_argument("--trigger", choices=("none", "Z"), default=None)
    p.add_argument("--trigger-ymax", type=float, default=None, dest="trigger_ymax", metavar="Y")
    p.add_argument("--lepton-ptmin", type=float, default=None, dest="lepton_ptmin", metavar="PT")
    p.add_argument("--lepton-etamax", type=float, default=None, dest="lepton_etamax", metavar="ETA")
    h = p.add_argument_group("hadrons", "Colorless on the final partons (run_prod_vacuum.py's code)")
    h.add_argument("--hadronize-xml", default=rj.HADRONIZE_XML, dest="hadronize_xml")
    h.add_argument("--hadronize-seed", type=int, default=1, dest="hadronize_seed")
    h.add_argument("--n-frag", type=int, default=1, dest="n_frag")
    h.add_argument("--hadron-keep-bits-p", type=int, default=None, dest="hadron_keep_bits_p")
    h.add_argument("--hadron-keep-bits-x", type=int, default=None, dest="hadron_keep_bits_x")
    h.add_argument("--eta-max", type=float, default=None, dest="eta_max")
    h.add_argument("--charged", action="store_true")
    h.add_argument("--no-x", action="store_true", dest="no_x")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    rp.add_seed_args(p, "Pythia, the initial state, MUSIC, Matter, LBT")
    rp.add_workdir_args(p)
    return p.parse_args()


def check_args(a) -> None:
    rv.check_args(a)                                 # sets a.hard, a.reuse, a.jets_per_bin
    a.reuse = a.reuse_req
    problems = []
    if a.reuse < 1:
        problems.append("--reuse must be >= 1")
    elif a.events % a.reuse:
        problems.append(f"--events {a.events} is not a multiple of --reuse {a.reuse}")
    for s in a.xml_set:
        if "=" not in s:
            problems.append(f"--xml-set {s!r}: need PATH=VALUE")
    if problems:
        sys.exit("run_prod_jet_single.py: cannot start:\n  " + "\n  ".join(problems))


def apply_xml_set(root, settings) -> None:
    for s in settings:
        path, value = s.split("=", 1)
        node = root.find(path.strip())
        if node is None:                             # create the missing elements
            node = root
            for tag in path.strip().split("/"):
                node = node.find(tag) if node.find(tag) is not None else ET.SubElement(node, tag)
        node.text = f" {value.strip()} "


def job_xml(a, stem: str):
    """The single-leg job XML: the user XML with --xml-set applied, its first <Hydro><MUSIC>
    only, no liquefier and no in-job hadronization -> (path, eCM, Eloss description)."""
    root = copy.deepcopy(ET.parse(a.user_xml).getroot())
    apply_xml_set(root, a.xml_set)
    problems = []
    for tag in DROP:
        for node in root.findall(tag):
            root.remove(node)
    hydros = root.findall("Hydro")
    musics = [h for h in hydros if h.find("MUSIC") is not None]
    if not musics:
        problems.append("need a <Hydro><MUSIC> block")
    for h in hydros:                                 # keep the first MUSIC (the background leg)
        if musics and h is not musics[0]:
            root.remove(h)
    if musics:
        hy = musics[0]
        rj._set(hy, "AddLiquefier", "false")
        m = hy.find("MUSIC")
        rj._set(m, "output_evolution_to_memory", 1)  # Matter and LBT read the medium
        rj._set(m, "dump_hydro_only", 0)
        rj._set(m, "freeze_out_surface", 0)          # no iSS here
        if rj._text(m, "skip_surface") is not None:
            rj._set(m, "skip_surface", 1)
    eloss = root.find("Eloss")
    if eloss is None or eloss.find("Matter") is None:
        problems.append("need <Eloss><Matter> (and usually <Lbt>)")
    else:
        rj._set(eloss, "AddLiquefier", "false")
        for mod in ("Matter", "Lbt"):
            if eloss.find(mod) is not None:
                rj._set(eloss.find(mod), "in_vac", 0)
    if root.find("IS") is None:
        problems.append("need <IS> (the medium's initial state and the hard vertex)")
    pg = root.find("Hard/PythiaGun")
    if pg is None:
        problems.append("need <Hard><PythiaGun>")
    else:
        if pg.find("extraHard") is not None:         # one hard scattering per event
            pg.remove(pg.find("extraHard"))
        problems += rj.configure_pythia_gun(a, pg)
    if problems:
        sys.exit(f"run_prod_jet_single.py: {a.user_xml}:\n  " + "\n  ".join(problems))
    rj._set(root, "nEvents", a.events)
    rj._set(root, "setReuseHydro", "true" if a.reuse > 1 else "false")
    rj._set(root, "nReuseHydro", a.reuse)
    rnd = root.find("Random") if root.find("Random") is not None else ET.SubElement(root, "Random")
    rj._set(rnd, "seed", a.seed)
    path = stem + ".xml"
    ET.ElementTree(root).write(path)
    desc = []
    for mod in ("Matter", "Lbt"):
        node = eloss.find(mod)
        if node is not None:
            desc.append(mod + " (" + ", ".join(f"{n.tag} {n.text.strip()}" for n in node
                                               if n.text and n.text.strip() and n.tag != "name") + ")")
    return path, float(rj._text(pg, "eCM")), "; ".join(desc)


def main() -> int:
    a = parse_args()
    a.reuse_req = a.reuse
    for k in ("build", "outdir", "main_xml", "user_xml", "hadronize_xml"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    check_args(a)
    os.makedirs(a.outdir, exist_ok=True)
    out_h5 = os.path.join(a.outdir, rp.output_name(a, "single_jet"))
    stem = os.path.splitext(out_h5)[0]
    rp.resolve_seed(a, out_h5, record=not a.dry_run)
    xml, ecm, eloss_desc = job_xml(a, stem)
    had_text, had_doc, had_was = rv.hadronize_blocks(a, ecm)
    root = ET.parse(xml).getroot()
    system = (f"{rj._text(root, 'IS/MCGlauber/projectile') or '?'}+"
              f"{rj._text(root, 'IS/MCGlauber/target') or '?'} {ecm:g} GeV, b in "
              f"[{rj._text(root, 'IS/MCGlauber/b_min') or 0}, {rj._text(root, 'IS/MCGlauber/b_max') or '?'}] fm")
    if a.pthat_windows:
        hard_desc = ("PythiaGun pTHat windows "
                     + ", ".join(f"{lo:g}-{hi:g}" for lo, hi in a.pthat_windows) + " GeV")
    else:
        hard_desc = (f"PythiaGun pTHat {rj._text(root, 'Hard/PythiaGun/pTHatMin')}-"
                     f"{rj._text(root, 'Hard/PythiaGun/pTHatMax')} GeV")
    if a.parton_ymax is not None:
        hard_desc += f", partons |y| < {a.parton_ymax:g} ({a.parton_y_mode})"
    if a.trigger_kind != "none":
        hard_desc += f", trigger {rj.trigger_desc(a)}"
    print(f"prod_single: {a.events} jet event(s), {a.events // a.reuse} hydro event(s) "
          f"(reuse {a.reuse}), seed {a.seed}, {system}, {hard_desc}")
    print(f"  build    {a.build}")
    print(f"  job XML  {xml}")
    print(f"  eloss    {eloss_desc}; no liquefier, one hydro leg")
    print(f"  hadrons  Colorless from {a.hadronize_xml}, eCMforHadronization {ecm:g}"
          + (f" (the file has {had_was})" if had_was and float(had_was) != ecm else ""))
    print(f"  output   {stem}_hadrons_jet_frag.h5")
    if a.dry_run:
        return 0

    import numpy as np
    import jetscape as js
    from jetscape import hadronizer as hz
    from jetscape import pyjetscape_core as core
    from jetscape.particlize_h5 import FINAL_PARTON_COLUMNS
    from jetscape.showers import initiators_from_manager

    file_uuid = str(uuidlib.uuid4())
    provenance = {
        "prod": "PyJetscape/example/prod_AuAu_0_10_jet/run_prod_jet_single.py",
        **rp.seed_provenance(a),
        "prod_host": socket.gethostname(),
        "prod_platform": platform.platform(),
        "prod_build": a.build,
        "prod_user_xml": open(xml).read(),
        "prod_template_xml": a.user_xml,
        "prod_xml_set": json.dumps(a.xml_set),
        "prod_hard": hard_desc,
        "system": system,
        "jet": f"{eloss_desc}; one hydro leg, no liquefier",
        "reuse": a.reuse,
        "vacuum": False,
        **({"pthat_bins": np.asarray(a.pthat_windows, dtype=np.float64)} if a.pthat_windows else {}),
        **({"parton_ymax": a.parton_ymax, "parton_y_mode": a.parton_y_mode}
           if a.parton_ymax is not None else {}),
        **({"trigger": a.trigger_kind,
            **{k: getattr(a, k) for k in rj.TRIGGER_CUTS if getattr(a, k) is not None}}
           if a.trigger_kind != "none" else {}),
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
        sys.exit(f"run_prod_jet_single.py: no PythiaGun task (tasks: {sorted(tasks)})")
    mgr = core.JetScapeSignalManager.Instance().GetJetEnergyLossManagerPointer()
    if mgr is None:
        sys.exit("run_prod_jet_single.py: no JetEnergyLossManager (no <Eloss> task?)")

    # Colorless outside the task list (run_prod_vacuum.py / run_prod_jet.py --hadrons-only)
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
    n, n_had, n_holes = 0, 0.0, 0
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
            holes = int((partons[:, 2] == -1).sum()) if len(partons) else 0
            n_holes += holes
            per_frag = sum(len(f["pid"]) for f in frags) / a.n_frag
            n_had += per_frag
            print(f"prod_single: event {i + 1}/{a.events} done in {time.time() - t0:.2f} s "
                  f"(hydro {i // a.reuse}), {len(ini)} showers, {len(partons)} final partons "
                  f"({holes} holes), {per_frag:.1f} particles/fragmentation, pTHat "
                  f"{d['pthat']:.1f} GeV (window {d['pthat_bin']})", flush=True)
        windows = rj.pthat_windows_final(gun)
        jetscape.Finish()
        finished = True
    finally:
        h.set_attrs({**(windows or {}), "prod_wall_s_total": time.time() - t_job,
                     "nevents": n, "complete_job": finished})
        h.close(complete=finished and n == a.events)

    summary = {"out": out_frag, "seed": a.seed, "seed_source": a.seed_source,
               "campaign": a.campaign, "index": a.index, "system": system, "hard": hard_desc,
               "jet": provenance["jet"], "reuse": a.reuse, "xml_set": a.xml_set,
               "events_requested": a.events, "events_written": n, "n_frag": a.n_frag,
               "particles_per_fragmentation": round(n_had / max(n, 1), 1),
               "holes_per_event": round(n_holes / max(n, 1), 1),
               "hadronize_base_seed": a.hadronize_seed, "file_uuid": file_uuid,
               "wall_s": round(time.time() - t_job, 1)}
    if windows is not None:
        summary.update(pthat_bins=windows["pthat_bins"].tolist(),
                       pthat_bin_sigma_gen_mb=list(windows["pthat_bin_sigma_gen"]),
                       pthat_bin_seeds=[int(x) for x in windows["pthat_bin_seeds"]])
    with open(stem + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print("prod_single:", json.dumps(summary))
    rp.leave_workdir(a, workdir, n == a.events)
    return 0 if n == a.events else 1


if __name__ == "__main__":
    raise SystemExit(main())
