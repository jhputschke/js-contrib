#!/bin/bash
# The HEPData records the bulk notebooks compare with, into ../runs/hepdata/ (git-ignored,
# shared with the Z+jet analyses' records, ../zjet/get_hepdata.sh):
#
#   HEPData-ins1666817-v1-yaml/   ALICE 1804.02944 (JHEP 07 (2018) 103): v_n{2}, v_n{4} of
#                                 charged hadrons in Pb-Pb at 5.02 and 2.76 TeV, |eta| < 0.8;
#                                 v2{2,|Delta eta|>1}(pT) 0-5 %, 5-10 % (v22Deta1_502TeVdif_cen0/1)
#   HEPData-ins1511868-v1-yaml/   CMS 1702.00630 (PLB 776 (2018) 195): v2{SP}(pT) up to 100 GeV
#                                 in Pb-Pb at 5.02 TeV, |eta| < 1, HF reference (|Delta eta| > 3);
#                                 0-5 %, 5-10 % (data1, data2)
#
#   ./get_hepdata.sh [alice|cms|all]   (default all; a record already there is kept)
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

if [[ $WHAT == alice || $WHAT == all ]]; then
  get ins1666817 1                                # HEPData ins1666817 v1 (doi:10.17182/hepdata.83737)
fi
if [[ $WHAT == cms || $WHAT == all ]]; then
  get ins1511868 1                                # HEPData ins1511868 v1 (doi:10.17182/hepdata.77603)
fi
