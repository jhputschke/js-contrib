# js-contrib tutorials

Step-by-step tutorials, one folder per contrib:

| folder | contrib | status |
|---|---|---|
| [`pyjetscape/`](pyjetscape/) | PyJetscape: running X-SCAPE from Python, the production scripts, the files, analysis in Python and ROOT, Python modules | in progress |

Each tutorial is a Jupyter notebook that runs on your own computer: in the production
container, with the xscape wheel on a Mac, or in the analysis environment. The analysis
tutorials also run in Google Colab. The website shows the notebooks with their outputs:
<https://jhputschke.github.io/js-contrib/tutorials/>.

## Getting them

In a checkout, the tutorials are already here. Without one, download a contrib's tutorials
with curl. A tutorial release (`tutorials-vN`) is pinned; the `main` branch is the latest
version:

```bash
# a release
curl -L https://github.com/jhputschke/js-contrib/releases/download/tutorials-v1/pyjetscape-tutorials.tar.gz | tar xz

# the main branch
mkdir pyjetscape-tutorials && curl -L https://codeload.github.com/jhputschke/js-contrib/tar.gz/refs/heads/main \
  | tar xz -C pyjetscape-tutorials --strip-components=3 js-contrib-main/tutorials/pyjetscape
```

Both commands create a folder `pyjetscape-tutorials/`. To start Jupyter in each setup, see
[`pyjetscape/README.md`](pyjetscape/README.md) or the
[setup page](https://jhputschke.github.io/js-contrib/tutorials/setup/).

## For authors

- **Folder layout.** A contrib's folder holds:
  - its notebooks (`NN_name.ipynb`, numbered in learning order)
  - `tutorials.json`, the index
  - `datasets.json`, the data the notebooks read
  - `_common/`, the helpers
  - `expected/`, the results the checks compare against

  Everything a tutorial needs is in the folder, so a tarball of it runs on its own.
- **Histograms.** Use [hist](https://hist.readthedocs.io), and
  [mplhep](https://mplhep.readthedocs.io) to draw them. Both are in every setup.
- **Releases.** `make_tarball.sh CONTRIB` packs a folder into
  `dist/<contrib>-tutorials.tar.gz`. Pushing a tag `tutorials-vN` publishes the tarballs
  as a GitHub release (`.github/workflows/tutorials.yml`). That release is never marked
  "latest", which belongs to js-contrib's code releases.
- **Website.** The site renders the notebooks with their stored outputs
  (`mkdocs-jupyter`; the hook `utils/mkdocs_hooks/tutorials.py` adds them to the site).
  Commit a notebook either without outputs, or executed with outputs that do not depend
  on your machine.
