# Building this site

The site is built with [MkDocs](https://www.mkdocs.org/) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme. Its pages are the Markdown files
in [`website/`](https://github.com/jhputschke/js-contrib/tree/main/website), and its
configuration, including the navigation, is in
[`mkdocs.yml`](https://github.com/jhputschke/js-contrib/blob/main/mkdocs.yml) at the
repository root. The `docs/` folder (design notes, plans, benchmarks) is not part of the site.

## Preview locally

```bash
python3 -m venv ~/.venvs/js_docs
~/.venvs/js_docs/bin/pip install -r website/requirements.txt
~/.venvs/js_docs/bin/mkdocs serve          # from the repository root
```

Then open <http://127.0.0.1:8000>. The page reloads whenever you save a file. To build the
static site into `_site/` and stop on any broken link or anchor:

```bash
~/.venvs/js_docs/bin/mkdocs build --strict
```

## Change a page

- Edit the Markdown file in `website/`. Each page also has an edit button that opens the file
  on GitHub.
- A new page also needs an entry under `nav:` in `mkdocs.yml`.
- Link to other pages with relative `.md` paths (`../run/osdf.md`). Link to files in the
  repository with full GitHub URLs (`https://github.com/jhputschke/js-contrib/blob/main/...`),
  because those files are not part of the site.
- Available features include admonitions (`!!! note`), content tabs (`=== "Docker"`), Mermaid
  diagrams, and math (`\( … \)`). See the
  [Material reference](https://squidfunk.github.io/mkdocs-material/reference/).

## Publish

Building the site does not publish anything. One way to put it on GitHub Pages at
`https://jhputschke.github.io/js-contrib/` is:

```bash
~/.venvs/js_docs/bin/mkdocs gh-deploy      # builds and pushes to the gh-pages branch
```

Then, in the repository's *Settings → Pages*, set the source to the `gh-pages` branch.
