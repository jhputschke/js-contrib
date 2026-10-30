#!/bin/bash
# CMS's HEPData records that the analyses compare with, into ../runs/hepdata/ (git-ignored,
# shared by every run; jetcorr.runs.hepdata):
#
#   HEPData-ins2945894-v1-yaml/   CMS 2507.09307 (CMS-HIN-23-006): Z-hadron Delta phi / Delta y
#                                 (mix.py, cms2507.py, test/compare.py, test/parton_cut_check.py)
#   HEPData-ins1512112-v1-yaml/   CMS 1702.01060 (CMS-HIN-15-013): Z+jet
#                                 (zjet_jets.py, test/zjet_pp_jets.py)
#   HEPData-ins3123406-v1-yaml/   CMS 2602.19431 (CMS-HIN-25-012): eta-separated dijet-hadron
#                                 correlations, the diffusion wake (../dijet/deta.py, cms25012.py)
#
#   ./get_hepdata.sh [2507|1702|25012|all]  (default all; a record already there is kept)
#   OUT=<dir> writes elsewhere than ../runs/hepdata/.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/../runs/hepdata}
WHAT=${1:-all}
mkdir -p "$OUT" && cd "$OUT"

get() {   # <inspire id> <version>
  local name=HEPData-$1-v$2-yaml
  if [[ -d $name ]]; then echo "$name: already in $OUT"; return; fi
  curl -sL "https://www.hepdata.net/download/submission/$1/$2/yaml" -o "$name.tar.gz"
  tar xzf "$name.tar.gz"                          # -> $name/
  echo "$name -> $OUT"
}

if [[ $WHAT == 2507 || $WHAT == all ]]; then
  # HEPData ins2945894 v1 (doi:10.17182/hepdata.156186), and the record's metadata
  get ins2945894 1
  [[ -f record_ins2945894.json ]] ||
    curl -sL "https://www.hepdata.net/record/156186?format=json&light=true" -o record_ins2945894.json
fi
if [[ $WHAT == 1702 || $WHAT == all ]]; then
  get ins1512112 1                                # HEPData ins1512112 v1
fi
if [[ $WHAT == 25012 || $WHAT == all ]]; then
  get ins3123406 1                                # HEPData ins3123406 v1 (doi:10.17182/hepdata.168219)
fi
