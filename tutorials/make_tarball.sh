#!/usr/bin/env bash
# make_tarball.sh CONTRIB [REF] -- pack tutorials/CONTRIB into dist/CONTRIB-tutorials.tar.gz
#
#   tutorials/make_tarball.sh pyjetscape          # from HEAD
#   tutorials/make_tarball.sh pyjetscape v1.4     # from a tag or branch
#
# The tracked files only (git archive), under one folder CONTRIB-tutorials/, so that
#   curl -L .../CONTRIB-tutorials.tar.gz | tar xz
# gives a folder that runs on its own.  .github/workflows/tutorials.yml attaches the tarballs
# to the release of a tag tutorials-vN.
set -euo pipefail
CONTRIB=${1:?usage: make_tarball.sh CONTRIB [REF]}
REF=${2:-HEAD}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$ROOT"
git cat-file -e "$REF:tutorials/$CONTRIB" 2>/dev/null \
  || { echo "make_tarball.sh: no tutorials/$CONTRIB at $REF (committed?)" >&2; exit 1; }
mkdir -p dist
OUT="dist/$CONTRIB-tutorials.tar.gz"
git archive --format=tar.gz --prefix="$CONTRIB-tutorials/" -o "$OUT" "$REF:tutorials/$CONTRIB"
echo "$OUT: $(tar tzf "$OUT" | wc -l | tr -d ' ') entries, $(du -h "$OUT" | cut -f1), version $(git show "$REF:tutorials/$CONTRIB/VERSION.txt")"
