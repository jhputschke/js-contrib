#!/usr/bin/env python3
"""A variant of the campaign's user XML with one setting changed, everything else byte for byte.

    python make_xml.py <campaign XML> <variant XML> --set Liquefier/CausalLiquefier/tau_delay=0.5

The setting is a path below <jetscape> to a leaf element.  Its tag must occur exactly once in
the file (tau_delay, e_threshold, ...; a tag that occurs more than once, like alphas, needs a
more specific edit): the value between its tags is replaced in the text, so the comments and
the layout stay, and the result is parsed back to check that the path now has the value.
check_settings.py inputs then compares the two files."""
import argparse
import re
import sys
import xml.etree.ElementTree as ET


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--set", dest="setting", required=True, help="PATH=VALUE")
    a = ap.parse_args()
    path, _, value = a.setting.partition("=")
    if not value:
        sys.exit("make_xml.py: --set PATH=VALUE")
    tag = path.split("/")[-1]
    text = open(a.src).read()
    if ET.fromstring(text).find(path) is None:
        sys.exit(f"make_xml.py: {path} not in {a.src}")
    pat = re.compile(rf"(<{re.escape(tag)}>)([^<]*)(</{re.escape(tag)}>)")
    hits = pat.findall(text)
    if len(hits) != 1:
        sys.exit(f"make_xml.py: <{tag}> occurs {len(hits)} times in {a.src}; one is needed")
    old = hits[0][1]
    pad_l = old[:len(old) - len(old.lstrip())]
    pad_r = old[len(old.rstrip()):]
    new = pat.sub(lambda m: f"{m.group(1)}{pad_l}{value}{pad_r}{m.group(3)}", text)
    got = ET.fromstring(new).find(path)
    if got is None or got.text.strip() != value:
        sys.exit(f"make_xml.py: {path} did not become {value}")
    header = (f"<!-- wake_scan/make_xml.py: {a.src} with {path} = {value} (was {old.strip()}) -->\n")
    if new.startswith("<?xml"):
        first, rest = new.split("\n", 1)
        new = first + "\n" + header + rest
    else:
        new = header + new
    with open(a.dst, "w") as f:
        f.write(new)
    print(f"{a.dst}: {path} {old.strip()} -> {value}")


if __name__ == "__main__":
    main()
