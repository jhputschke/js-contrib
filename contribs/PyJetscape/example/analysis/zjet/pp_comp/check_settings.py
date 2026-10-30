#!/usr/bin/env python3
"""The pp comparison's settings: the PP19 sample must differ from the campaign's in Matter's
Q0 alone.

    python check_settings.py inputs <campaign user XML> <PP19 user XML>
    python check_settings.py jobs <campaign job XML> <PP19 job XML>

inputs: the two user XMLs' <Hard><PythiaGun> blocks are identical (LinesToRead line by line)
and their <Eloss><Matter> blocks agree in everything run_prod_vacuum.py does not force
(in_vac, recoil_on) but Q0.  jobs: the job XMLs run_prod_vacuum.py wrote (<stem>.xml) differ in
<Eloss><Matter><Q0> and nothing else.  Prints the differences; exits 1 on any other."""
import sys
import xml.etree.ElementTree as ET

ALLOWED = {"Eloss/Matter/Q0"}
FORCED = {"in_vac", "recoil_on"}        # run_prod_vacuum.py sets them in both


def norm(text):
    if text is None:
        return ""
    lines = [" ".join(x.split()) for x in text.strip().splitlines()]
    return "\n".join(x for x in lines if x)


def flat(node, path=""):
    """{path: text} of every element with text, the leaves of the tree."""
    out = {}
    for child in node:
        if not isinstance(child.tag, str):          # comments
            continue
        p = f"{path}/{child.tag}" if path else child.tag
        if len(child):
            out.update(flat(child, p))
        else:
            out[p] = norm(child.text)
    return out


def diffs(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("inputs", "jobs"):
        sys.exit(__doc__)
    mode, pa, pb = sys.argv[1:]
    ra, rb = ET.parse(pa).getroot(), ET.parse(pb).getroot()
    if mode == "inputs":
        a = {**{f"Hard/PythiaGun/{k}": v for k, v in flat(ra.find("Hard/PythiaGun")).items()},
             **{f"Eloss/Matter/{k}": v for k, v in flat(ra.find("Eloss/Matter")).items()
                if k not in FORCED}}
        b = {**{f"Hard/PythiaGun/{k}": v for k, v in flat(rb.find("Hard/PythiaGun")).items()},
             **{f"Eloss/Matter/{k}": v for k, v in flat(rb.find("Eloss/Matter")).items()
                if k not in FORCED}}
    else:
        a, b = flat(ra), flat(rb)
    d = diffs(a, b)
    for k in d:
        print(f"  {k}: {a.get(k, '(absent)')!r} -> {b.get(k, '(absent)')!r}")
    bad = [k for k in d if k not in ALLOWED]
    if bad:
        print(f"check_settings {mode}: FAILED, differences beyond Q0: {bad}")
        sys.exit(1)
    if not d:
        print(f"check_settings {mode}: FAILED, no difference at all (Q0 should differ)")
        sys.exit(1)
    print(f"check_settings {mode}: OK, only Matter Q0 differs")


if __name__ == "__main__":
    main()
