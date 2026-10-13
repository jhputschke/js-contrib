#!/usr/bin/env python3
"""
docs/analysis_snippets/run_doc_blocks.py -- run the Python code blocks of a guide.

How the blocks marked *(tested)* in README_analysis.md and README_OO.md are checked: the
```python blocks of the Markdown file run one after the other in one namespace (later blocks
use what earlier ones defined), with the guide's placeholder paths replaced by real ones.

    # README_OO.md on a copy of an O+O campaign (its ROOT block writes next to the inputs)
    python docs/analysis_snippets/run_doc_blocks.py docs/README_OO.md --tested \\
        --sub /path/to/OO_c1=/scratch/oo_copy

    # README_analysis.md on the seed-1 files of prod_AuAu_0_10_jet/out
    O=contribs/PyJetscape/example/prod_AuAu_0_10_jet/out
    python docs/analysis_snippets/run_doc_blocks.py docs/README_analysis.md --tested \\
        --sub /path/to/campaign=$O --sub AuAu_0_10_jet_c1_0001=$O/AuAu_0_10_jet_seed0001 \\
        --sub pth10-40_c1_campaign.root=$O/campaign_campaign.root

    python docs/analysis_snippets/run_doc_blocks.py docs/README_OO.md --list    # the blocks

Run it from the js-contrib root (the guides' relative paths start there), with PyJetscape
importable (the analysis venv, or js_fno after `conda activate`).  --tested runs only the
blocks whose preceding text says "(tested"; the others may need data the test files lack
(pT-hat windows, for example).  The first failing block stops the run with its traceback.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def blocks(text):
    """-> [(index, line number, tested, code)] for every ```python block."""
    out, prev_end = [], 0
    for i, m in enumerate(BLOCK.finditer(text)):
        before = text[prev_end:m.start()]
        line = text.count("\n", 0, m.start()) + 1
        # the marker is in the heading or sentence just before the block
        out.append((i, line, "(tested" in before[-700:], m.group(1)))
        prev_end = m.end()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc", help="a Markdown guide")
    ap.add_argument("--sub", action="append", default=[], metavar="OLD=NEW",
                    help="replace OLD by NEW in every block (repeatable; longest OLD first)")
    ap.add_argument("--tested", action="store_true",
                    help="only the blocks marked (tested) in the text before them")
    ap.add_argument("--list", action="store_true", help="list the blocks and stop")
    a = ap.parse_args(argv)

    text = open(a.doc).read()
    found = blocks(text)
    if a.list:
        for i, line, tested, code in found:
            first = code.strip().splitlines()[0] if code.strip() else ""
            print(f"{i:2d}  line {line:4d}  {'tested' if tested else '      '}  {first[:70]}")
        return 0
    subs = []
    for s in a.sub:
        if "=" not in s:
            ap.error(f"--sub {s!r}: OLD=NEW")
        old, new = s.split("=", 1)
        subs.append((old, new))
    subs.sort(key=lambda x: -len(x[0]))

    ns = {"__name__": "__doc_blocks__"}
    ran = 0
    for i, line, tested, code in found:
        if a.tested and not tested:
            continue
        for old, new in subs:
            code = code.replace(old, new)
        t0 = time.time()
        print(f"--- block {i} (line {line})", flush=True)
        exec(compile(code, f"{a.doc}:block{i}:line{line}", "exec"), ns)
        print(f"--- block {i} ok, {time.time() - t0:.1f} s", flush=True)
        ran += 1
    print(f"run_doc_blocks: {ran} block(s) of {a.doc} ran")
    return 0


if __name__ == "__main__":
    sys.exit(main())
