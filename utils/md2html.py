#!/usr/bin/env python3
"""A Markdown note (an analysis summary, a README) as one self-contained HTML page, for
reading in a browser or sending around: tables, figures embedded as data URIs, heading
anchors, simple inline math.

    python utils/md2html.py runs/zjet_c00-01/figures/ANALYSIS_c00_01.md     # -> .html next to it
    python utils/md2html.py NOTE.md -o out.html --title "My note"

When the .html exists already (e.g. made by pandoc), its <head> is kept, so the page keeps
its title and style; otherwise a built-in style is used, the title from ``--title`` or the
first heading.  Figures: relative .png / .jpg / .gif / .svg paths are embedded (absolute URLs
stay links).  Inline math ``$...$`` outside code: letters in italics, Greek macros (``\\mu``,
``\\Delta``, ...) as their letters; anything else is left as written, with a warning.
Needs markdown-it-py (it comes with rich, in the analysis environment)."""
import argparse
import base64
import html
import os
import re
import sys

from markdown_it import MarkdownIt

STYLE = """body { max-width: 1100px; margin: 2em auto; padding: 0 16px; font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #222; background: #fff; }
h1, h2 { border-bottom: 1px solid #ddd; padding-bottom: .2em; }
table { border-collapse: collapse; margin: 1em 0; font-size: 14px; display: block; overflow-x: auto; }
th, td { border: 1px solid #ccc; padding: 4px 8px; vertical-align: top; }
th { background: #f3f3f3; }
img { max-width: 100%; height: auto; border: 1px solid #eee; }
code { background: #f5f5f5; padding: 0 3px; border-radius: 3px; font-size: 90%; white-space: pre-wrap; }
pre { background: #f5f5f5; padding: 10px; overflow-x: auto; }
pre code { background: none; padding: 0; white-space: pre; }"""

GREEK = {n: chr(c) for n, c in (
    ("alpha", 945), ("beta", 946), ("gamma", 947), ("delta", 948), ("epsilon", 949),
    ("zeta", 950), ("eta", 951), ("theta", 952), ("iota", 953), ("kappa", 954),
    ("lambda", 955), ("mu", 956), ("nu", 957), ("xi", 958), ("pi", 960), ("rho", 961),
    ("sigma", 963), ("tau", 964), ("upsilon", 965), ("phi", 966), ("chi", 967), ("psi", 968),
    ("omega", 969), ("Gamma", 915), ("Delta", 916), ("Theta", 920), ("Lambda", 923),
    ("Xi", 926), ("Pi", 928), ("Sigma", 931), ("Phi", 934), ("Psi", 936), ("Omega", 937))}
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
        ".svg": "image/svg+xml"}


def math(text):
    """Inline $...$ outside code -> italics with Greek letters; unhandled ones kept."""
    def conv(m):
        tex = m.group(1)
        out = re.sub(r"\\([A-Za-z]+)", lambda g: GREEK.get(g.group(1), g.group(0)), tex)
        if re.search(r"[\\{}^_]", out):
            print(f"md2html: math left as written: ${tex}$", file=sys.stderr)
            return m.group(0)
        return f"<em>{html.escape(out)}</em>"
    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S)
    for i in range(0, len(parts), 2):              # the even parts are outside code
        parts[i] = re.sub(r"(?<![\\\w$])\$([^$\s][^$\n]*?)(?<!\s)\$(?![\w$])", conv, parts[i])
    return "".join(parts)


def slug(s):
    s = re.sub(r"<[^>]+>", "", html.unescape(s)).lower()
    s = re.sub(r"[^\w\s.-]", "", s)
    return re.sub(r"\s+", "-", s.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("note", help="the Markdown file")
    ap.add_argument("-o", "--output", help="the HTML file (default: the note's name, .html)")
    ap.add_argument("--title", help="the page title (a new page; default: the first heading)")
    a = ap.parse_args()
    dst = a.output or os.path.splitext(a.note)[0] + ".html"
    here = os.path.dirname(os.path.abspath(a.note))
    text = open(a.note, encoding="utf-8").read()

    body = MarkdownIt("commonmark", {"html": True}).enable("table").render(math(text))
    seen = {}

    def heading(m):
        sid = slug(m.group(2))
        n = seen.get(sid, 0)
        seen[sid] = n + 1
        sid = sid if n == 0 else f"{sid}-{n}"
        return f'<h{m.group(1)} id="{sid}">{m.group(2)}</h{m.group(1)}>'

    body = re.sub(r"<h([1-6])>(.*?)</h\1>", heading, body, flags=re.S)
    missing = []

    def embed(m):
        src = html.unescape(m.group(1))
        path = os.path.join(here, src)
        mime = MIME.get(os.path.splitext(src)[1].lower())
        if mime is None or not os.path.exists(path):
            missing.append(src)
            return m.group(0)
        data = base64.b64encode(open(path, "rb").read()).decode()
        return f'<img src="data:{mime};base64,{data}"'

    body, n_img = re.subn(r'<img src="(?![a-z]+:)([^"]+)"', embed, body)
    for src in missing:
        print(f"md2html: figure not embedded (missing or not an image): {src}", file=sys.stderr)

    head = None
    if os.path.exists(dst):
        old = open(dst, encoding="utf-8").read()
        if "<body>" in old:
            head = old[:old.index("<body>") + len("<body>")]
    if head is None:
        title = a.title
        if title is None:
            h = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
            title = re.sub(r"<[^>]+>", "", html.unescape(h.group(1))).strip() if h else a.note
        head = (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8" />\n'
                f'<meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
                f"<title>{html.escape(title)}</title>\n<style>\n{STYLE}\n</style>\n</head>\n"
                f"<body>")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(head + "\n" + body + "</body>\n</html>\n")
    print(f"{dst}: {n_img - len(missing)} figures embedded, {body.count('<table>')} tables, "
          f"{len(seen)} headings, {os.path.getsize(dst) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
