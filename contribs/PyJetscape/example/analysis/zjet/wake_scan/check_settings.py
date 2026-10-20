#!/usr/bin/env python3
"""Two XMLs must differ in the scanned setting alone.

    python check_settings.py <XML a> <XML b> --allow Liquefier/CausalLiquefier/tau_delay \\
        [--allow Random/seed]

Every leaf below <jetscape> is compared (text, whitespace normalized; comments ignored).  The
differences are printed; exit 1 if one is not in --allow, or if none of the allowed ones
differs while they were expected to (--expect-diff, the default for user XMLs and same-seed job
XMLs of two variants).  Use it on the user XMLs (make_xml.py's output against the campaign's)
and on the job XMLs run_prod_jet.py writes (<stem>.xml), seed by seed."""
import argparse
import sys
import xml.etree.ElementTree as ET


def norm(text):
    lines = [" ".join(x.split()) for x in (text or "").strip().splitlines()]
    return "\n".join(x for x in lines if x)


def flat(node, path=""):
    out = {}
    for child in node:
        if not isinstance(child.tag, str):
            continue
        p = f"{path}/{child.tag}" if path else child.tag
        if len(child):
            out.update(flat(child, p))
        else:
            out[p] = norm(child.text)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--allow", action="append", required=True)
    ap.add_argument("--no-expect-diff", dest="expect", action="store_false")
    x = ap.parse_args()
    fa, fb = flat(ET.parse(x.a).getroot()), flat(ET.parse(x.b).getroot())
    diffs = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    for k in diffs:
        print(f"  {k}: {fa.get(k, '(absent)')!r} -> {fb.get(k, '(absent)')!r}")
    bad = [k for k in diffs if k not in x.allow]
    if bad:
        print(f"check_settings: FAILED, other differences: {bad}")
        sys.exit(1)
    if x.expect and not diffs:
        print("check_settings: FAILED, the scanned setting does not differ")
        sys.exit(1)
    print(f"check_settings: OK ({', '.join(diffs) if diffs else 'identical'})")


if __name__ == "__main__":
    main()
