#!/usr/bin/env python3
"""
analysis/OO-RHIC/rhictune_xml.py -- the full-chain p+p XML of the RHIC tune, made from the O+O one.

    python rhictune_xml.py write     # config/OO_MCGlauber_MUSIC_isr_rhictune.xml -> config/pp200_isr_rhictune.xml
    python rhictune_xml.py check     # exit 1 if the two differ anywhere but in the nuclei

config/pp200_isr_rhictune.xml is p+p 200 GeV in the same chain as O+O (3dMCGlauber strings,
PythiaIsrGun + iMATTER, MUSIC, iSS, Matter/LBT, iColored): the p+p reference of the physics run and
the p+p underlying-event samples.  It is not edited by hand.  It is written from the O+O file with
only the nuclei changed (PP_TAGS: projectile and target p, no O16 configuration file) and its own
header; every other element is the O+O file's, so the two chains cannot drift apart.  `check`
compares the element trees (tags, attributes, stripped texts; comments ignored) and lists every
difference outside PP_TAGS; oo_bias.py runs it before it launches jobs with either file.

Until 2026-11-24 the p+p samples ran the O+O file with the same three changes as driver options
(oo_bias.py: --xml-set=IS/MCGlauber/projectile=p etc.); the job XMLs of both ways are the same
(checked with the driver's dry run), so those samples count as made with this file.
config/pp200_isr_vacuum.xml is a different chain: the hard tune's, without strings and hydro.
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / "config"
OO = CONFIG / "OO_MCGlauber_MUSIC_isr_rhictune.xml"
PP = CONFIG / "pp200_isr_rhictune.xml"
#: the only elements in which the p+p file differs from the O+O file: path -> p+p value
PP_TAGS = {"IS/MCGlauber/projectile": "p", "IS/MCGlauber/target": "p",
           "IS/MCGlauber/nucleon_configuration_from_file": "0"}

HEADER = """<!--
  p+p 200 GeV in the full chain of the RHIC tune: 3dMCGlauber strings -> PythiaIsrGun + iMATTER
  (ISR) -> MUSIC -> iSS, jet in Matter + LBT, iColored hadronization.  The p+p reference of the
  O+O physics run (analysis/OO-RHIC oo_bias.py, sample pp_ref_w*) and the chain of the p+p
  underlying-event samples (pp_bulk_*), driven by prod_OO_isr/run_prod_oo_isr.py (option user-xml).

  GENERATED, do not edit: analysis/OO-RHIC/rhictune_xml.py writes it from
  OO_MCGlauber_MUSIC_isr_rhictune.xml with only the nuclei changed (projectile and target p,
  nucleon_configuration_from_file 0; b_max 3 fm as in the O+O file, overwritten per job by the
  driver).  Every other setting, and the reasons for it (the hard tune, the soft MPI, the bulk
  tune, tau_form), is the O+O file's: see its header and comments, which are kept below.  Edit the
  O+O file and run `python rhictune_xml.py write`; `python rhictune_xml.py check` (also run by
  oo_bias.py before launching) fails if the two files differ anywhere else.

  Not to be confused with pp200_isr_vacuum.xml: the hard tune's chain (no strings, no hydro, Matter
  in vacuum), with the same Pythia and Matter settings.
-->"""


def _flat(root):
    """{path: (attrib, stripped text)} for every element; repeated tags get an index."""
    out = {}

    def walk(el, path):
        seen = {}
        for ch in el:
            if not isinstance(ch.tag, str):
                continue
            seen[ch.tag] = seen.get(ch.tag, 0) + 1
            p = f"{path}/{ch.tag}" if path else ch.tag
            key = p if seen[ch.tag] == 1 else f"{p}[{seen[ch.tag]}]"
            out[key] = (dict(ch.attrib), (ch.text or "").strip())
            walk(ch, key)
    walk(root, "")
    return out


def differences(a=OO, b=PP):
    """Element paths where a and b differ, other than PP_TAGS with b holding the p+p value."""
    fa, fb = _flat(ET.parse(a).getroot()), _flat(ET.parse(b).getroot())
    bad = []
    for k in sorted(set(fa) | set(fb)):
        if k in PP_TAGS:
            if k not in fb or fb[k][1] != PP_TAGS[k]:
                bad.append(f"{k}: p+p file has {fb.get(k, ('', '<missing>'))[1]!r}, expected {PP_TAGS[k]!r}")
            continue
        if fa.get(k) != fb.get(k):
            bad.append(f"{k}: {fa.get(k, '<missing>')} vs {fb.get(k, '<missing>')}")
    return bad


def write():
    s = OO.read_text()
    # the O+O description comment (the first comment after the licence lines) -> the p+p header,
    # the O+O one kept right after it as the reference for every setting
    m = re.search(r"<!--\n  O\+O 200 GeV with initial-state radiation.*?-->", s, re.S)
    if not m:
        sys.exit("rhictune_xml.py: the O+O description comment not found")
    oo_doc = m.group(0).replace("<!--\n", "<!-- The O+O file's description (all settings below are its own):\n", 1)
    s = s[:m.start()] + HEADER + "\n\n" + oo_doc + s[m.end():]
    for path, val in PP_TAGS.items():
        tag = path.rsplit("/", 1)[1]
        s, n = re.subn(rf"<{tag}>[^<]*</{tag}>", f"<{tag}>{val}</{tag}>", s, count=1)
        if n != 1:
            sys.exit(f"rhictune_xml.py: <{tag}> not found once in {OO.name}")
    PP.write_text(s)
    bad = differences()
    if bad:
        sys.exit("rhictune_xml.py: written, but the check fails:\n  " + "\n  ".join(bad))
    print(f"wrote {PP} (differs from {OO.name} only in {', '.join(PP_TAGS)})")


def check(quiet=False):
    bad = differences()
    if bad:
        print(f"rhictune_xml.py: {PP.name} differs from {OO.name} beyond the nuclei "
              "(run `python rhictune_xml.py write`):\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    if not quiet:
        print(f"{PP.name} = {OO.name} except {', '.join(PP_TAGS)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "write":
        write()
    elif cmd == "check":
        sys.exit(check())
    else:
        sys.exit(__doc__)
