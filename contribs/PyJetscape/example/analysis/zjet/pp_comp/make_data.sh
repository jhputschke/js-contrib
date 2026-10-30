#!/usr/bin/env bash
# analysis/zjet/pp_comp/make_data.sh: the pp comparison (README.md), from the production to the
# figures.  Two pp Z+jet samples with the same seed (the same Pythia events, paired):
#   campaign  ../../../config/PbPb_0_30_ZJet.xml, the campaign's pp reference (Matter Q0 = 2 GeV)
#   pp19      ZJet_pp_PP19.xml, JETSCAPE's PP19 tune (Matter Q0 = 1 GeV), all else the same
# both with the campaign's arguments (utils/zjet/README.md, "pp reference").
#
#   ./make_data.sh [all | prod | check | map | pools | compare]      (default: all)
#
#   prod     run_prod_vacuum.py for both samples (in parallel, CPU), then check
#   check    the job XMLs differ in Matter's Q0 alone (check_settings.py jobs)
#   map      zjet_jets_map.py (R = 0.3) per sample -> data/jets/<sample>/
#   pools    mix.py --pools-only (the Z-hadron pools) per sample -> data/mix/<sample>/vacuum/
#   compare  compare_pp.py -> figures/pp_comp_*.png, pp_comp_summary.txt
#
# Environment: OUT (default ../../runs/pp_comp), EVENTS (12000), SEED (900001), NFRAG (40);
# needs X-SCAPE's build_gpu with the PyJetscape bindings and the HEPData records
# (../get_hepdata.sh).  Outputs are git-ignored (runs/): the results are not public.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ZJET=$(dirname "$HERE")
ANA=$(dirname "$ZJET")
EX=$(dirname "$ANA")
OUT=${OUT:-$ANA/runs/pp_comp}
DATA=$OUT/data
LOGS=$DATA/logs
EVENTS=${EVENTS:-12000}
SEED=${SEED:-900001}
NFRAG=${NFRAG:-40}
WINDOWS=20-30,30-35,35-45,45-55,55-70,70-200
CAMPAIGN_XML=$EX/config/PbPb_0_30_ZJet.xml
PP19_XML=$HERE/ZJet_pp_PP19.xml
SAMPLES="campaign pp19"

xml_of() { case $1 in campaign) echo "$CAMPAIGN_XML" ;; pp19) echo "$PP19_XML" ;; esac; }
name_of() { case $1 in campaign) echo zjet_pp ;; pp19) echo zjet_pp_pp19 ;; esac; }

check() {
  python "$HERE/check_settings.py" jobs "$DATA"/campaign/*_seed*.xml "$DATA"/pp19/*_seed*.xml
}

prod() {
  python "$HERE/check_settings.py" inputs "$CAMPAIGN_XML" "$PP19_XML"
  mkdir -p "$LOGS"
  for s in $SAMPLES; do
    echo "prod $s: $EVENTS events, seed $SEED, $NFRAG fragmentations -> $DATA/$s"
    python "$EX/prod_AuAu_0_10_jet/run_prod_vacuum.py" --events "$EVENTS" --seed "$SEED" \
      --campaign "$(name_of $s)" --user-xml "$(xml_of $s)" --pthat-bins "$WINDOWS" \
      --lepton-ptmin 20 --lepton-etamax 2.5 --n-frag "$NFRAG" --eta-max 2.6 --no-x \
      --hadron-keep-bits-p 12 --outdir "$DATA/$s" > "$LOGS/prod_$s.log" 2>&1 &
  done
  wait
  check
}

map() {
  mkdir -p "$LOGS"
  for s in $SAMPLES; do
    echo "map $s -> $DATA/jets/$s"
    python "$ZJET/zjet_jets_map.py" "$DATA/$s" "$DATA/jets/$s" > "$LOGS/map_$s.log" 2>&1 &
  done
  wait
}

pools() {
  mkdir -p "$LOGS"
  for s in $SAMPLES; do
    echo "pools $s -> $DATA/mix/$s/vacuum"
    python "$ZJET/mix.py" --pbpb none --vacuum "$DATA/$s" --pools "$DATA/mix/$s" \
      --pools-only --jobs 1 > "$LOGS/pools_$s.log" 2>&1 &
  done
  wait
}

compare() {
  python "$HERE/compare_pp.py" --data "$DATA" --outdir "$OUT/figures"
}

step=${1:-all}
case $step in
  all) prod; map; pools; compare ;;
  prod | check | map | pools | compare) $step ;;
  *) echo "usage: $0 [all | prod | check | map | pools | compare]" >&2; exit 2 ;;
esac
