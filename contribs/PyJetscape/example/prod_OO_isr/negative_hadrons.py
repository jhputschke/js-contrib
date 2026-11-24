#!/usr/bin/env python3
"""
example/prod_OO_isr/negative_hadrons.py

The medium's holes as negative hadrons, for a run_prod_oo_isr.py job made with
--store-partons --store-hadronization-input (the jet in the medium: Matter with recoil_on 1
and/or LBT).

Why.  An elastic scattering in the medium knocks a thermal parton out (the recoil, pstat 1)
and leaves a hole (pstat -1): the four-momentum the medium lost.  The hydro does not feed
back, so energy is conserved only as recoil - hole.  iColoredHadronization hadronizes the
recoils but drops the holes, and for a hole whose colour line is still open after its
reconnection it puts a colour partner into the string system, with the hole's
three-momentum: per event (recoil + partner) instead of (recoil - hole).  This script
hadronizes, per event, the holes plus a copy of every partner in a separate string system
and writes its hadrons with pstat -1: an analysis counts them with weight -1 (jets: subtract
the negative hadrons in the cone), so the event's balance becomes
(recoil + partner) - (hole + partner) = recoil - hole, as in JETSCAPE's
ColorlessHadronization (take_recoil).

    conda activate js_fno
    python negative_hadrons.py OUT/OO_isr_seed0001_hadrons_jet_frag.h5
                        # -> OUT/OO_isr_seed0001_hadrons_jet_neg.h5
    python negative_hadrons.py F.h5 --no-partners      # the holes alone
    python negative_hadrons.py F.h5 --soft-energy 0.5  # another energy of the soft ends

The negative system (one per event; all holes together, robust against small masses):
  - its partons: the holes (partons/, pstat -1) and a copy of each partner (the
    hadronization_input/ entry with the hole's three-momentum); their colour tags are not
    used (they do not form a singlet), the strings are built anew;
  - quarks and antiquarks are paired by the smallest angle, each pair a string; every gluon
    joins the string whose ends are closest in angle, ordered by its angle to the quark end;
    gluons without any quark form a closed gluon loop (two or more), ordered in azimuth;
  - unpaired quark ends get soft end partons (d or dbar, --soft-energy each, default
    0.2 GeV) whose momentum is carved out of the system: each takes the fraction
    soft-energy/E of the system's four-momentum, the other partons are scaled by the rest,
    so the system's four-momentum stays exactly holes + partner copies;
  - Pythia (iColoredHadronization's settings and decays, from the jet_frag file) fragments
    it; four-momentum is conserved per system, so the negative hadrons sum exactly to the
    holes + partner copies.
  - A system that cannot fragment (a single parton, invariant mass below --min-mass,
    default 1 GeV, soft ends taking more than half its energy, or a failed fragmentation)
    is written as its negative partons (pstat -2; neg_mode 1, 2 for a failure): soft, exact,
    subtracted at parton level.

Output <stem>_hadrons_jet_neg.h5: jetscape.hadrons_h5 jet_frag layout, one sample per unit,
the jet_frag file's units (unit, event, seed of this pass), hadrons pstat -1, and per unit
neg_n_holes, neg_hole_E, neg_n_partners, neg_partner_E, neg_n_soft, neg_soft_E,
neg_in_E/px/py/pz (holes + partner copies) and neg_out_E/px/py/pz (what is written),
neg_mode (0 hadrons, 1 partons: too small to fragment, 2 partons: fragmentation failed).  Model
results, not vetted.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("_rehad", os.path.join(HERE, "rehadronize_jets.py"))
RH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RH)                  # rp (paths), user_xml
rp = RH.rp

QUARKS = (1, 2, 3, 4, 5)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jet_frag", help="a <stem>_hadrons_jet_frag.h5 with partons/ and hadronization_input/")
    p.add_argument("--out", default=None, help="default: <stem>_hadrons_jet_neg.h5 next to the input")
    p.add_argument("--no-partners", action="store_true", dest="no_partners",
                   help="the holes alone (no partner copies)")
    p.add_argument("--soft-energy", type=float, default=0.2, dest="soft_energy",
                   help="energy of the soft end partons [GeV] (default 0.2)")
    p.add_argument("--min-mass", type=float, default=1.0, dest="min_mass",
                   help="smallest invariant mass of a system that is fragmented [GeV] (default 1)")
    p.add_argument("--seed-base", type=int, default=1, dest="seed_base",
                   help="per-event seeds from (BASE, unit) (default 1)")
    p.add_argument("--events", default=None, help="unit range a:b (default all)")
    p.add_argument("--build", default=rp.BUILD)
    p.add_argument("--main-xml", default=rp.MAIN_XML, dest="main_xml")
    p.add_argument("--workdir", default=None)
    p.add_argument("--force", action="store_true", help="overwrite --out")
    return p.parse_args(argv)


def neg_seed(base, unit):
    s = np.random.SeedSequence([int(base), 7, int(unit)])
    return int(s.generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF) or 1


def negative_partons(partons, hin, with_partners=True):
    """The event's holes and the copies of their partners: (pid, p (K, 4) [E, px, py, pz],
    mass) and the counts/energies."""
    holes = partons[partons[:, 2] == -1] if partons is not None and len(partons) else np.zeros((0, 14))
    pid = [int(h[1]) for h in holes]
    p4 = [h[3:7] for h in holes]
    n_par, e_par = 0, 0.0
    if with_partners and hin is not None and len(holes):
        hp = np.asarray(hin["p"], float)
        for h in holes:
            j = np.flatnonzero(np.abs(hp[:, 1:4] - h[4:7]).sum(1) < 1e-6)
            if len(j):
                k = int(j[0])
                pid.append(int(hin["pid"][k]))
                p4.append(hp[k])
                n_par += 1
                e_par += float(hp[k, 0])
    p4 = np.asarray(p4, float).reshape(-1, 4)
    return (np.asarray(pid, int), p4, len(holes), float(holes[:, 3].sum()) if len(holes) else 0.0,
            n_par, e_par)


def _angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.pi
    return float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1)))


def build_system(pid, p4, e_soft):
    """Colour tags for a singlet system of the given partons, plus soft end partons for
    unpaired quark ends, carved out of the system (its four-momentum is unchanged).
    Returns pid, col, acol, p4, mass, n_soft, soft_E; None if the soft ends would take more
    than half the system's energy."""
    pid = list(pid)
    p4 = [np.asarray(x, float) for x in p4]
    P = np.sum(p4, axis=0)
    q = [i for i, f in enumerate(pid) if f in QUARKS]
    qb = [i for i, f in enumerate(pid) if -f in QUARKS]
    g = [i for i, f in enumerate(pid) if f == 21]
    n_soft = abs(len(q) - len(qb))
    f_soft = e_soft / P[0] if P[0] > 0 else 1.0
    if n_soft * f_soft > 0.5:
        return None
    if n_soft:                                   # carve the soft ends out of the system
        scale = 1.0 - n_soft * f_soft
        p4 = [x * scale for x in p4]
    while len(q) > len(qb):
        pid.append(-1); p4.append(P * f_soft); qb.append(len(pid) - 1)
    while len(qb) > len(q):
        pid.append(1); p4.append(P * f_soft); q.append(len(pid) - 1)
    col = [0] * len(pid)
    acol = [0] * len(pid)
    tag = 600

    def new():
        nonlocal tag
        tag += 1
        return tag

    if not q:                                    # closed gluon loop, ordered in azimuth
        order = sorted(g, key=lambda i: np.arctan2(p4[i][2], p4[i][1]))
        tags = [new() for _ in order]
        for k, i in enumerate(order):
            col[i], acol[i] = tags[k], tags[k - 1]
    else:
        pairs, free = [], list(qb)               # quark - antiquark by the smallest angle
        for i in q:
            j = min(free, key=lambda j: _angle(p4[i][1:4], p4[j][1:4]))
            free.remove(j)
            pairs.append([i, [], j])
        for k in g:                              # every gluon to the closest string
            best = min(pairs, key=lambda s: min(_angle(p4[k][1:4], p4[s[0]][1:4]),
                                                _angle(p4[k][1:4], p4[s[2]][1:4])))
            best[1].append(k)
        for i, gs, j in pairs:
            gs.sort(key=lambda k: _angle(p4[k][1:4], p4[i][1:4]))
            prev = new()
            col[i] = prev
            for k in gs:
                acol[k] = prev
                prev = new()
                col[k] = prev
            acol[j] = prev
    p4 = np.asarray(p4, float).reshape(-1, 4)
    mass = np.sqrt(np.clip(p4[:, 0] ** 2 - (p4[:, 1:4] ** 2).sum(1), 0, None))
    return (np.asarray(pid, np.int32), np.asarray(col, np.int32), np.asarray(acol, np.int32), p4,
            mass, n_soft, n_soft * e_soft)


def invariant_mass(p4):
    P = np.asarray(p4, float).reshape(-1, 4).sum(0)
    return float(np.sqrt(max(P[0] ** 2 - (P[1:4] ** 2).sum(), 0.0)))


def main(argv=None):
    a = parse_args(argv)
    for k in ("jet_frag", "build", "main_xml"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    from jetscape import pyjetscape_core as core
    from jetscape.hadrons_h5 import HadronFile, HadronH5Writer
    if not getattr(core, "ICOLORED_HAS_INPUT_RECORD", False):
        sys.exit("negative_hadrons.py: this PyJetscape has no hadronize_input")
    hf = HadronFile(a.jet_frag)
    if hf.tag != "jet_frag" or not hf.has_partons:
        sys.exit(f"negative_hadrons.py: {a.jet_frag} has no partons/ (run_prod_oo_isr.py --store-partons)")
    if not a.no_partners and not hf.has_hadronization_input:
        sys.exit(f"negative_hadrons.py: {a.jet_frag} has no hadronization_input/ "
                 "(--store-hadronization-input), needed for the partners; or --no-partners")
    jh_xml = str(hf.attrs.get("jet_hadronization_xml", ""))
    if not jh_xml:
        sys.exit(f"negative_hadrons.py: {a.jet_frag} has no jet_hadronization_xml")
    out = os.path.abspath(a.out or a.jet_frag.replace("_hadrons_jet_frag.h5", "_hadrons_jet_neg.h5"))
    if out == a.jet_frag:
        sys.exit("negative_hadrons.py: --out would overwrite the input")
    if os.path.exists(out) and not a.force:
        sys.exit(f"negative_hadrons.py: {out} exists (use --force)")
    units = list(range(hf.n_units))
    if a.events:
        units = units[slice(*[int(x) if x else None for x in a.events.split(":")])]
    workdir = os.path.abspath(a.workdir or os.path.join(os.path.dirname(out), "work_negative",
                                                        os.path.basename(out).replace(".h5", "")))
    os.makedirs(workdir, exist_ok=True)
    job_xml = os.path.join(workdir, "negative.xml")
    ET.ElementTree(RH.user_xml(jh_xml, None)).write(job_xml)
    main_xml = os.path.join(workdir, "jetscape_main.xml")
    rp._absolute_parent_paths(a.main_xml, main_xml, a.build)
    cwd = os.getcwd()
    os.chdir(workdir)
    writer, finished = None, False
    n_fail = n_part = 0
    max_dev = 0.0
    try:
        jetscape = core.JetScapePerEvent()
        jetscape.SetXMLMainFileName(main_xml)
        jetscape.SetXMLUserFileName(job_xml)
        jetscape.Init()
        module = core.create_module("iColoredHadronization")
        module.Init()
        writer = HadronH5Writer(
            out, tag="jet_frag", n_samples=1,
            attrs={**{k: v for k, v in hf.attrs.items()
                      if k in ("prod", "prod_seed", "system", "prod_hard", "eos", "eloss",
                               "subtract_hard_momentum", "source", "prod_user_xml",
                               "hadrons_only", "jet_hadronization_xml", "pythia_version")},
                   "module": "iColoredHadronization (negative_hadrons.py)",
                   "negative": True, "negative_of": os.path.basename(a.jet_frag),
                   "partners": not a.no_partners, "soft_energy": a.soft_energy,
                   "seed_base": a.seed_base,
                   "method": "holes + partner copies, one string system per event, pstat -1"})
        for u in units:
            unit = int(hf.units["unit"][u]) if "unit" in hf.units else u
            parts = hf.partons(unit)
            hin = hf.hadronization_input(unit) if not a.no_partners else None
            pid, p4, nh, eh, npar, epar = negative_partons(parts, hin, not a.no_partners)
            seed = neg_seed(a.seed_base, unit)
            rec = {"neg_n_holes": nh, "neg_hole_E": eh, "neg_n_partners": npar,
                   "neg_partner_E": epar, "neg_n_soft": 0, "neg_soft_E": 0.0, "neg_mode": 0}
            d = {"pid": np.zeros(0, np.int32), "pstat": np.zeros(0, np.int32),
                 "p": np.zeros((0, 4)), "x": np.zeros((0, 4))}
            Pin = p4.sum(0) if len(pid) else np.zeros(4)
            if len(pid):
                sys_ = (build_system(pid, p4, a.soft_energy)
                        if len(pid) > 1 and invariant_mass(p4) >= a.min_mass else None)
                had = None
                if sys_ is not None:
                    sp, sc, sa, sp4, sm, ns, esoft = sys_
                    rec.update(neg_n_soft=ns, neg_soft_E=esoft)
                    try:
                        had = core.hadronize_input(module, sp, sc, sa, sp4, sm, seed)
                    except RuntimeError:
                        n_fail += 1
                if had is not None:
                    d = dict(had)
                    d["pstat"] = np.full(len(d["pid"]), -1, np.int32)
                else:                            # the negative partons themselves, exact
                    n_part += 1
                    rec["neg_mode"] = 1 if sys_ is None else 2
                    d = {"pid": pid.astype(np.int32), "pstat": np.full(len(pid), -2, np.int32),
                         "p": p4, "x": np.zeros((len(pid), 4))}
            Pout = np.asarray(d["p"], float).reshape(-1, 4).sum(0) if len(d["pid"]) else np.zeros(4)
            max_dev = max(max_dev, float(np.abs(Pout - Pin).max()))
            rec.update({f"neg_in_{c}": float(v) for c, v in zip(("E", "px", "py", "pz"), Pin)})
            rec.update({f"neg_out_{c}": float(v) for c, v in zip(("E", "px", "py", "pz"), Pout)})
            writer.append_unit([{k: d[k] for k in ("pid", "pstat", "p", "x") if k in d}], unit=unit,
                               event=int(hf.units.get("event", units)[u]), seed=seed, **rec)
        jetscape.Finish()
        finished = True
    finally:
        if writer is not None:
            writer.close(complete=finished)
        hf.close()
        os.chdir(cwd)
        shutil.rmtree(workdir, ignore_errors=True)
        try:
            os.rmdir(os.path.dirname(workdir))
        except OSError:
            pass
    print(f"negative_hadrons.py: {len(units)} event(s), {n_part} written as negative partons "
          f"(pstat -2; {n_fail} of them after a failed fragmentation); max |sum p(out) - sum p(holes + partners)| "
          f"{max_dev:.2e} GeV -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
