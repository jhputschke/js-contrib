"""Prepare a page's HTML for the Markdown copies behind /llms.txt (plugin mkdocs-llmstxt).

The plugin's own cleaning drops the labels of content tabs, so the commands of the "Docker",
"Apptainer", "Mac" and "Source build" tabs would follow each other without saying which is
which. mkdocs.yml therefore sets `autoclean: false`, and this module does the cleaning:

- content tabs: each tab's label becomes a bold line in front of its content;
- notebooks (mkdocs-jupyter): the prompts ("In [3]:") and the copy buttons, which repeat every
  code cell as plain text, are removed; code cells are marked as Python;
- math: `\\( ... \\)` and `\\[ ... \\]` become code spans holding `$ ... $`, so the TeX
  survives the conversion to Markdown unescaped;
- Mermaid diagrams are marked as such;
- then the plugin's cleaning (images, permalinks, emoji).
"""
import re

from bs4 import NavigableString
from mkdocs_llmstxt import autoclean

_MATH = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)


def _tabs(soup):
    for tabs in soup.find_all(class_="tabbed-set"):
        labels = tabs.find(class_="tabbed-labels", recursive=False)
        content = tabs.find(class_="tabbed-content", recursive=False)
        if labels is None or content is None:
            continue
        names = [label.get_text(" ", strip=True) for label in labels.find_all("label")]
        blocks = content.find_all(class_="tabbed-block", recursive=False)
        for name, block in zip(names, blocks):
            line = soup.new_tag("p")
            bold = soup.new_tag("strong")
            bold.string = f"{name}:"
            line.append(bold)
            block.insert(0, line)


def _notebooks(soup):
    for cls in ("jp-InputPrompt", "jp-OutputPrompt", "zeroclipboard-container",
                "clipboard-copy-txt"):
        for element in soup.find_all(class_=cls):
            element.decompose()
    for element in soup.find_all("a", class_="anchor-link"):
        element.decompose()
    for block in soup.find_all(class_="highlight-ipynb"):
        for pre in block.find_all("pre"):
            pre["class"] = [*(pre.get("class") or []), "language-python"]


def _math(soup):
    for node in list(soup.find_all(string=_MATH)):
        if node.find_parent(["pre", "code", "script", "style"]) is not None:
            continue
        text, parts, last = str(node), [], 0
        for found in _MATH.finditer(text):
            parts.append(NavigableString(text[last:found.start()]))
            code = soup.new_tag("code")
            code.string = "$" + " ".join((found.group(1) or found.group(2)).split()) + "$"
            parts.append(code)
            last = found.end()
        parts.append(NavigableString(text[last:]))
        node.replace_with(*parts)


def preprocess(soup, output):          # the plugin calls this for every page it lists
    _tabs(soup)
    _notebooks(soup)
    _math(soup)
    for pre in soup.find_all("pre", class_="mermaid"):
        pre["class"] = [*pre["class"], "language-mermaid"]
    autoclean(soup)
