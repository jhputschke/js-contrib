#!/usr/bin/env bash
# analysis/zjet/wake_scan/make_data.sh: a paired scan of one liquefier setting (README.md), from
# the variant XMLs to the figures.  Default: CausalLiquefier tau_delay = 0.5, 1.0 (the campaign's,
# the baseline), 2.0, each with the GB10 mini campaign's four campaign-format jobs (seeds
# 102-105, 48 events, two hydro backgrounds each): the same events, backgrounds, jets and
# droplets in every value, only the jet leg's fluid (MUSIC_2) differs.
#
#   ./make_data.sh [all | xml | prod | check | pools | compare | pp | iso] (default: all)
#
#   PARAM=Liquefier/e_threshold VALUES="1.0 2.0 3.0" BASE=2.0 REF=tau_delay_1p0 PAIRING=events \
#     ./make_data.sh                  # the campaign's value from an earlier scan, not rerun
#   VARIANT=mpi_off LINES="PartonLevel:MPI = off" PARAM=Liquefier/e_threshold VALUES="2.0 3.0" \
#     BASE=2.0 PAIRING=events ./make_data.sh prod pp iso     # a scan with other Pythia settings
#
#   xml      the variant XMLs from the campaign's (make_xml.py), each checked against it
#   prod     run_prod_jet.py for every value and seed, PAR jobs at a time (GPU), then check;
#            jobs whose <stem>.json exists are skipped (a rerun resumes)
#   check    per seed, every value against the baseline: the job XMLs differ in the setting
#            alone (check_settings.py), the jet files share events, partons and droplets
#            (check_pairing.py)
#   pools    mix.py --pools-only per value -> data/mix/<name>/pbpb/ (version 4); a VARIANT's pp
#            too (data/mix/pp_<VARIANT>/vacuum/)
#   compare  compare_scan.py -> figures/wake_scan_<setting>.png, .txt
#   pp       a VARIANT's own pp reference (run_prod_vacuum.py, the variant XML, pp_comp's
#            arguments: PP_EVENTS, PP_SEED) -> data/pp_<VARIANT>/
#   iso      the wake and the fragments on their own (docs/Plans/PLAN_wake_isolated.md, step 1):
#            ../wake_iso.py map per value and for the pp (PP_HADRONS) -> data/iso/<name>.npz
#            (cached), then compare_iso.py -> figures/wake_iso_*.png, wake_iso_<setting>.txt
#
# Environment: OUT (../../runs/wake_scan), PARAM (Liquefier/CausalLiquefier/tau_delay),
# VALUES ("0.5 1.0 2.0"), BASE (1.0), SEEDS ("102 103 104 105"), EVENTS (48), PAR (2),
# REF (the run name whose jobs and pools are the baseline, e.g. tau_delay_1p0: an earlier
# scan's campaign-setting jobs with the same seeds; check verifies that its job XMLs differ
# from the other values' in PARAM alone), PAIRING (jets: the setting acts after the energy
# loss, the jets must be identical; events: it acts during it, e_threshold, alpha_s, ...),
# PP_POOLS (the campaign's pp pools, ../../runs/pp_comp/data/mix/campaign/vacuum: made by
# ../pp_comp/make_data.sh prod pools), PP_HADRONS (the pp hadron files for iso: the campaign's
# ../../runs/pp_comp/data/campaign, a VARIANT's data/pp_<VARIANT>).
# Other Pythia settings: VARIANT (a prefix of the run names, e.g. mpi_off), LINES (lines of
# PythiaGun's <LinesToRead> to set, separated by ";", e.g. "PartonLevel:ISR = off;
# PartonLevel:MPI = off"; make_xml.py --line), WINDOWS (the pTHat windows,
# 20-30,30-35,35-45,45-55,55-70,70-200), JPB (jets per window per background, 4; windows x JPB
# = 24 keeps two backgrounds per job), PP_EVENTS (12000), PP_SEED (900001).  A VARIANT's values
# pair among themselves (the same Pythia events), not with the campaign's runs.  Outputs are
# git-ignored (runs/).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ZJET=$(dirname "$HERE")
ANA=$(dirname "$ZJET")
EX=$(dirname "$ANA")
PROD=$EX/prod_AuAu_0_10_jet
CONFIG=$EX/config
OUT=${OUT:-$ANA/runs/wake_scan}
DATA=$OUT/data
LOGS=$DATA/logs
PARAM=${PARAM:-Liquefier/CausalLiquefier/tau_delay}
VALUES=${VALUES:-"0.5 1.0 2.0"}
BASE=${BASE:-1.0}
SEEDS=${SEEDS:-"102 103 104 105"}
REF=${REF:-}
PAIRING=${PAIRING:-jets}
EVENTS=${EVENTS:-48}
PAR=${PAR:-2}
PP_POOLS=${PP_POOLS:-$ANA/runs/pp_comp/data/mix/campaign/vacuum}
VARIANT=${VARIANT:-}
LINES=${LINES:-}
WINDOWS=${WINDOWS:-20-30,30-35,35-45,45-55,55-70,70-200}
JPB=${JPB:-4}
PP_EVENTS=${PP_EVENTS:-12000}
PP_SEED=${PP_SEED:-900001}
if [[ -n $VARIANT ]]; then PP_DEFAULT=$DATA/pp_$VARIANT; else PP_DEFAULT=$ANA/runs/pp_comp/data/campaign; fi
PP_HADRONS=${PP_HADRONS:-$PP_DEFAULT}
CAMPAIGN_XML=$CONFIG/PbPb_0_30_ZJet.xml
TAG=${PARAM##*/}

name_of() {        # a value's run name; the baseline's is REF when given
  if [[ -n $REF && $1 == "$BASE" ]]; then echo "$REF"; else echo "${VARIANT:+${VARIANT}_}${TAG}_${1//./p}"; fi
}
line_args() {      # LINES as make_xml.py --line arguments, one per line of output
  local IFS=';'
  for l in $LINES; do printf -- '--line\n%s\n' "$(echo "$l" | sed 's/^ *//; s/ *$//')"; done
}
skip_base() { [[ -n $REF && $1 == "$BASE" ]]; }
stem_of() { printf 'PbPb_0_30_zjet_%s_seed%04d' "$1" "$2"; }

xml() {
  mkdir -p "$DATA/xml"
  for v in $VALUES; do
    skip_base "$v" && continue
    local n; n=$(name_of "$v")
    local -a extra=()
    mapfile -t extra < <(line_args)
    python "$HERE/make_xml.py" "$CAMPAIGN_XML" "$DATA/xml/$n.xml" --set "$PARAM=$v" \
      "${extra[@]}"
    python "$HERE/check_settings.py" "$CAMPAIGN_XML" "$DATA/xml/$n.xml" --allow "$PARAM" \
      ${LINES:+--allow Hard/PythiaGun/LinesToRead} --no-expect-diff
  done
}

run_job() {        # value seed: one campaign-format job, the mini campaign's arguments
  local n stem
  n=$(name_of "$1")
  stem=$(stem_of "$n" "$2")
  mkdir -p "$DATA/$n" "$LOGS"
  if [[ -f $DATA/$n/$stem.json ]]; then
    echo "  $stem: done before"
    return 0
  fi
  echo "  $stem"
  (cd "$PROD" && MUSIC_CUDA_SYNC=block OMP_NUM_THREADS=6 python run_prod_jet.py \
    --events "$EVENTS" --seed "$2" --out "$stem.h5" --user-xml "$DATA/xml/$n.xml" \
    --grid "$CONFIG/grid_PbPb.yaml" --seed-registry none --pthat-bins "$WINDOWS" \
    --jets-per-bin "$JPB" --lepton-ptmin 20 --lepton-etamax 2.5 --hadrons-only --oversample 40 \
    --n-frag 40 --correlated --eta-max 2.6 --no-x --hadron-keep-bits-p 12 \
    --hadron-keep-bits-x 8 --keep-primaries 0 --outdir "$DATA/$n") \
    > "$LOGS/prod_$stem.log" 2>&1 || echo "  $stem FAILED (see $LOGS/prod_$stem.log)"
}

prod() {
  xml
  local i=0
  echo "prod: $PARAM = $VALUES, seeds $SEEDS, $EVENTS events each, $PAR at a time"
  for s in $SEEDS; do              # seed by seed: each round gives same-seed pairs early
    for v in $VALUES; do
      skip_base "$v" && continue
      run_job "$v" "$s" &
      i=$((i + 1))
      if ((i % PAR == 0)); then wait; fi
    done
  done
  wait
  check
}

check() {
  local bn fail=0
  bn=$(name_of "$BASE")
  for v in $VALUES; do
    [[ $v == "$BASE" ]] && continue
    local n; n=$(name_of "$v")
    for s in $SEEDS; do
      local b c
      b=$(stem_of "$bn" "$s")
      c=$(stem_of "$n" "$s")
      python "$HERE/check_settings.py" "$DATA/$bn/$b.xml" "$DATA/$n/$c.xml" --allow "$PARAM" \
        || fail=1
      python "$HERE/check_pairing.py" "$DATA/$bn/${b}_hadrons_jet_frag.h5" \
        "$DATA/$n/${c}_hadrons_jet_frag.h5" --level "$PAIRING" || fail=1
    done
  done
  if ((fail)); then echo "check: FAILED"; return 1; fi
  echo "check: every value paired with $BASE"
}

pools() {
  mkdir -p "$LOGS"
  for v in $VALUES; do
    local n; n=$(name_of "$v")
    if skip_base "$v" && ls "$DATA/mix/$n/pbpb"/*.npz > /dev/null 2>&1; then
      echo "pools $n: there (REF)"
      continue
    fi
    echo "pools $n -> $DATA/mix/$n/pbpb"
    python "$ZJET/mix.py" --pbpb "$DATA/$n" --vacuum none --pools "$DATA/mix/$n" --pools-only \
      --jobs 4 > "$LOGS/pools_$n.log" 2>&1
  done
  if [[ -n $VARIANT ]] && ls "$PP_HADRONS"/*_hadrons_jet_frag.h5 > /dev/null 2>&1; then
    echo "pools pp_$VARIANT -> $DATA/mix/pp_$VARIANT/vacuum"
    python "$ZJET/mix.py" --pbpb none --vacuum "$PP_HADRONS" --pools "$DATA/mix/pp_$VARIANT" \
      --pools-only --jobs 1 > "$LOGS/pools_pp_$VARIANT.log" 2>&1
  fi
}

compare() {
  if ! ls "$PP_POOLS"/*.npz > /dev/null 2>&1 && ! ls "$PP_POOLS"/*_pool.root > /dev/null 2>&1; then
    echo "no pp pools in $PP_POOLS: run ../pp_comp/make_data.sh prod pools, or set PP_POOLS" >&2
    return 1
  fi
  # shellcheck disable=SC2086
  python "$HERE/compare_scan.py" --data "$DATA" --pp "$PP_POOLS" --param "$PARAM" \
    --values $VALUES --base "$BASE" ${REF:+--base-name "$REF"} --outdir "$OUT/figures"
}

iso() {
  if ! ls "$PP_HADRONS"/*_hadrons_jet_frag.h5 > /dev/null 2>&1; then
    echo "no pp hadron files in $PP_HADRONS: run ../pp_comp/make_data.sh prod, or set PP_HADRONS" >&2
    return 1
  fi
  mkdir -p "$DATA/iso" "$LOGS"
  for v in $VALUES; do
    local n; n=$(name_of "$v")
    echo "iso map $n -> $DATA/iso/$n.npz"
    python "$ZJET/wake_iso.py" map --src "$DATA/$n" --out "$DATA/iso/$n.npz" --jobs 4 \
      > "$LOGS/iso_$n.log" 2>&1
  done
  local ppn=pp_${VARIANT:-campaign}
  echo "iso map pp -> $DATA/iso/$ppn.npz"
  python "$ZJET/wake_iso.py" map --src "$PP_HADRONS" --out "$DATA/iso/$ppn.npz" \
    --jobs 1 > "$LOGS/iso_$ppn.log" 2>&1
  # shellcheck disable=SC2086
  python "$HERE/compare_iso.py" --data "$DATA" --param "$PARAM" --values $VALUES \
    --base "$BASE" ${REF:+--base-name "$REF"} ${VARIANT:+--variant "$VARIANT"} \
    --pp "$DATA/iso/$ppn.npz" --outdir "$OUT/figures"
}

pp() {             # a VARIANT's pp reference: its Pythia settings in vacuum
  if [[ -z $VARIANT ]]; then
    echo "pp: the campaign's pp reference is ../pp_comp's (make_data.sh prod there)" >&2
    return 1
  fi
  local n out
  n=$(name_of "$BASE")
  out=$DATA/pp_$VARIANT
  mkdir -p "$LOGS"
  [[ -f $DATA/xml/$n.xml ]] || xml
  if ls "$out"/*_seed*.json > /dev/null 2>&1; then
    echo "pp $VARIANT: done before ($out)"
    return 0
  fi
  echo "pp $VARIANT: $PP_EVENTS events, seed $PP_SEED -> $out"
  python "$PROD/run_prod_vacuum.py" --events "$PP_EVENTS" --seed "$PP_SEED" \
    --campaign "zjet_pp_$VARIANT" --user-xml "$DATA/xml/$n.xml" --pthat-bins "$WINDOWS" \
    --lepton-ptmin 20 --lepton-etamax 2.5 --n-frag 40 --eta-max 2.6 --no-x \
    --hadron-keep-bits-p 12 --outdir "$out" > "$LOGS/pp_$VARIANT.log" 2>&1
}

for step in "${@:-all}"; do
  case $step in
    all) prod; pools; compare; iso ;;
    xml | prod | check | pools | compare | pp | iso) $step ;;
    *) echo "usage: $0 [all | xml | prod | check | pools | compare | pp | iso] ..." >&2; exit 2 ;;
  esac
done
