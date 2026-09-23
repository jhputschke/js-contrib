# Vendoring `fast_data`

`python/fast_data/` is a **verbatim copy** of `loc_libs/fast_data/` from the FNO4d
repository. The exact upstream commit and a per-file sha256 live in
[`python/fast_data/UPSTREAM.json`](python/fast_data/UPSTREAM.json), and
`tests/test_vendor_intact.py` fails if anything drifts.

## The rule

**No local patches to `python/fast_data/`.** Ever.

Every adaptation the JETSCAPE side needs goes in `python/fasthydro/` instead. This is what
makes re-syncing a mechanical operation rather than a merge. If you find yourself wanting to
edit a vendored file, that is a signal the change belongs upstream in FNO4d — make it there,
then re-sync.

The one thing that would justify an entry in `UPSTREAM.json`'s `local_patches` list is a
change that cannot be expressed upstream (a hard dependency that X-SCAPE cannot carry, say).
Record the reason there if it ever happens.

## Why vendor at all

js-contrib must be usable without an FNO4d checkout. The tree is small — 19 files, ~380 KB —
so the cost of carrying it is far below the cost of a cross-repo runtime dependency.

**This contribution is the home of `fast_data` for X-SCAPE purposes.** FNO4d keeps its copy
for training-data work; when the two diverge, upstream FNO4d wins and this copy is re-synced.

## Re-syncing

```bash
tools/sync_fast_data.sh [path-to-FNO4d]      # default: ~/FNO4d
```

It rsyncs the tree, rewrites `UPSTREAM.json` with the new commit and checksums, and prints
`git diff --stat` so the change is reviewed rather than rubber-stamped. Afterwards:

```bash
pytest tests/test_vendor_intact.py tests/test_fast_data_*.py
```

## Name collision

`fast_data` is a top-level package name that also exists in FNO4d. If both are importable,
whichever comes first on `sys.path` wins, silently. `fasthydro/__init__.py` therefore checks
that the imported `fast_data` really is the vendored one and raises naming both paths if not.
Set `FASTHYDRO_ALLOW_EXTERNAL_FAST_DATA=1` to override deliberately.
