#!/usr/bin/env python3
"""A variant of the campaign's user XML with one setting changed, everything else byte for byte.

    python make_xml.py <campaign XML> <variant XML> --set Liquefier/CausalLiquefier/tau_delay=0.5
    python make_xml.py <campaign XML> <variant XML> --set Liquefier/e_threshold=3.0 \
        --line "PartonLevel:MPI = off" [--line "PartonLevel:ISR = off"]

The setting is a path below <jetscape> to a leaf element.  Its tag must occur exactly once in
the file (tau_delay, e_threshold, ...; a tag that occurs more than once, like alphas, needs a
more specific edit): the value between its tags is replaced in the text, so the comments and
the layout stay, and the result is parsed back to check that the path now has the value.
--line "KEY = VALUE" changes one line of a block of Pythia settings (PythiaGun's
<LinesToRead>): the line "KEY = <old>" must occur exactly once, and only its value changes.
check_settings.py inputs then compares the two files."""
import argparse
import re
import sys
import xml.etree.ElementTree as ET


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--set", dest="setting", default=None, help="PATH=VALUE")
    ap.add_argument("--line", action="append", default=[],
                    help='"KEY = VALUE": one line of a Pythia settings block (repeatable)')
    a = ap.parse_args()
    if a.setting is None and not a.line:
        sys.exit("make_xml.py: nothing to change (--set PATH=VALUE and/or --line 'KEY = VALUE')")
    text = open(a.src).read()
    changes = []
    if a.setting is not None:
        text, change = set_leaf(text, a.setting, a.src)
        changes.append(change)
    for line in a.line:
        text, change = set_line(text, line, a.src)
        changes.append(change)
    header = (f"<!-- wake_scan/make_xml.py: {a.src} with {'; '.join(changes)} -->\n")
    if text.startswith("<?xml"):
        first, rest = text.split("\n", 1)
        text = first + "\n" + header + rest
    else:
        text = header + text
    with open(a.dst, "w") as f:
        f.write(text)
    print(f"{a.dst}: {'; '.join(changes)}")


def set_leaf(text, setting, src):
    """The leaf PATH=VALUE replaced in the text -> (text, description)."""
    path, _, value = setting.partition("=")
    if not value:
        sys.exit("make_xml.py: --set PATH=VALUE")
    tag = path.split("/")[-1]
    if ET.fromstring(text).find(path) is None:
        sys.exit(f"make_xml.py: {path} not in {src}")
    pat = re.compile(rf"(<{re.escape(tag)}>)([^<]*)(</{re.escape(tag)}>)")
    hits = pat.findall(text)
    if len(hits) != 1:
        sys.exit(f"make_xml.py: <{tag}> occurs {len(hits)} times in {src}; one is needed")
    old = hits[0][1]
    pad_l = old[:len(old) - len(old.lstrip())]
    pad_r = old[len(old.rstrip()):]
    new = pat.sub(lambda m: f"{m.group(1)}{pad_l}{value}{pad_r}{m.group(3)}", text)
    got = ET.fromstring(new).find(path)
    if got is None or got.text.strip() != value:
        sys.exit(f"make_xml.py: {path} did not become {value}")
    return new, f"{path} = {value} (was {old.strip()})"


def set_line(text, line, src):
    """The line "KEY = <old>" of a settings block set to "KEY = VALUE" -> (text, description)."""
    key, _, value = (x.strip() for x in line.partition("="))
    if not key or not value:
        sys.exit("make_xml.py: --line 'KEY = VALUE'")
    pat = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)(\S[^\n<]*?)(\s*)$", re.M)
    hits = pat.findall(text)
    if len(hits) != 1:
        sys.exit(f"make_xml.py: the line '{key} = ...' occurs {len(hits)} times in {src}; one "
                 "is needed")
    old = hits[0][1]
    new = pat.sub(lambda m: f"{m.group(1)}{value}{m.group(3)}", text)
    ET.fromstring(new)                                           # still well formed
    if len(pat.findall(new)) != 1 or pat.findall(new)[0][1] != value:
        sys.exit(f"make_xml.py: '{key}' did not become {value}")
    return new, f"{key} = {value} (was {old})"


if __name__ == "__main__":
    main()
