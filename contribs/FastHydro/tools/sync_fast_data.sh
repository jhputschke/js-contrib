#!/usr/bin/env bash
# Re-vendor python/fast_data/ from an FNO4d checkout and refresh UPSTREAM.json.
#
#   tools/sync_fast_data.sh [path-to-FNO4d]
#
# See VENDORING.md.  Hard rule: no local patches to the vendored tree.
set -euo pipefail

FNO4D="${1:-$HOME/FNO4d}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$FNO4D/loc_libs/fast_data"
DST="$HERE/python/fast_data"

[[ -d "$SRC" ]] || { echo "error: $SRC does not exist (pass the FNO4d path as \$1)" >&2; exit 1; }

COMMIT=$(git -C "$FNO4D" log -1 --format=%H -- loc_libs/fast_data)
HEAD_AT=$(git -C "$FNO4D" rev-parse HEAD)
CDATE=$(git -C "$FNO4D" log -1 --format=%ad --date=short -- loc_libs/fast_data)

if ! git -C "$FNO4D" diff --quiet -- loc_libs/fast_data; then
    echo "warning: $SRC has uncommitted changes; the recorded commit will not describe it" >&2
fi

echo "==> rsync $SRC -> $DST"
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='UPSTREAM.json' \
      "$SRC/" "$DST/"

echo "==> rewriting UPSTREAM.json  (commit ${COMMIT:0:7}, $CDATE)"
COMMIT="$COMMIT" HEAD_AT="$HEAD_AT" CDATE="$CDATE" DST="$DST" python3 - <<'PY'
import hashlib, json, os, pathlib, datetime
root = pathlib.Path(os.environ["DST"])
files = {}
for f in sorted(root.rglob("*")):
    if f.is_file() and f.name != "UPSTREAM.json" and "__pycache__" not in f.parts:
        b = f.read_bytes()
        files[str(f.relative_to(root))] = {"sha256": hashlib.sha256(b).hexdigest(),
                                           "bytes": len(b), "lines": b.count(b"\n")}
old = root / "UPSTREAM.json"
patches = json.loads(old.read_text())["local_patches"] if old.exists() else []
(root / "UPSTREAM.json").write_text(json.dumps({
    "_comment": "Provenance of the vendored fast_data tree. Verified by tests/test_vendor_intact.py. "
                "Hard rule: no local patches -- every adapter need goes in python/fasthydro/.",
    "upstream": {"repo": "FNO4d", "path": "loc_libs/fast_data",
                 "commit": os.environ["COMMIT"], "commit_short": os.environ["COMMIT"][:7],
                 "commit_date": os.environ["CDATE"],
                 "repo_head_at_vendoring": os.environ["HEAD_AT"]},
    "vendored_on": datetime.date.today().isoformat(),
    "local_patches": patches,
    "files": files,
}, indent=2) + "\n")
print(f"    {len(files)} files")
PY

echo "==> review this before committing:"
git -C "$HERE" diff --stat -- python/fast_data || true
echo
echo "then: pytest tests/test_vendor_intact.py tests/test_fast_data_*.py"
