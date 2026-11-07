#!/usr/bin/env bash
# make_data.sh -- the liquefier frame test's data: the same 25 Au+Au jet events in the three
# frames of the CausalLiquefier (<CausalLiquefier><frame> lab, bjorken, fluid), each with its
# hydro pair, particlize file and hadrons; then the per-event summary the notebook reads.
#
#   ./make_data.sh                         # all three frames, then the summary
#   OUT=/data/lft ./make_data.sh fluid     # one frame (the summary needs all three)
#   ./make_data.sh summary                 # only the summary (frame_summary.py)
#
# The events are those of tutorial 06 (tutorials/v2/AuAu_pair_mini, data_recipes/make_v2.sh
# pair_mini), extended from 5 to 25 jets: 0-10% Au+Au at 200 GeV, one background (seed 101)
# reused for every jet, PythiaGun pTHat 30-40 GeV with both hard partons at |y| < 0.6, both
# freeze-out surfaces; hadronized with correlated sampling (200 samples per event and for the
# background, sample k of each event paired with sample k of the background) and 20
# fragmentations per event.  The first 5 events are the tutorial's (same seed and settings);
# only the liquefier's frame differs between the three runs: the background, the jets and the
# droplets are bit-identical (frame_summary.py checks it).
#
# Needs an X-SCAPE with <CausalLiquefier><frame>fluid</frame> (branch liquefier-rest-frame)
# and its pyjetscape_core.  About 15 min on an M3 Max with the three frames in parallel
# (JOBS=3), ~7 GB per frame.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PROD="$HERE/../prod_AuAu_0_10_jet"
OUT=${OUT:-$HOME/JetScape/liquefier_frame_test}
PY=${PY:-python}
EVENTS=${EVENTS:-25}
SEED=${SEED:-101}
OVERSAMPLE=${OVERSAMPLE:-200}
JOBS=${JOBS:-3}                            # frames in parallel
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-5} OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
STEM=$(printf "AuAu_0_10_jet_seed%04d" "$SEED")

log() { echo "[$(date +%H:%M:%S)] $*"; }

one_frame() {
  local fr=$1 D="$OUT/$1"
  mkdir -p "$D"
  log "frame $fr -> $D"
  (cd "$PROD" &&
   $PY run_prod_jet.py --events "$EVENTS" --reuse "$EVENTS" --seed "$SEED" \
       --pthat-min 30 --pthat-max 40 --parton-ymax 0.6 --write-particlize both \
       --liquefier-frame "$fr" --outdir "$D" > "$D/prod.log" 2>&1 &&
   $PY hadronize.py "$D/${STEM}_particlize.h5" \
       --correlated --common-seeds --oversample "$OVERSAMPLE" --n-frag 20 > "$D/hadronize.log" 2>&1)
  log "frame $fr done ($(du -sh "$D" | cut -f1))"
}

summary() {
  log "summary -> $OUT/summary.npz"
  $PY "$HERE/frame_summary.py" "$OUT" -o "$OUT/summary.npz"
}

ARGS=${*:-lab bjorken fluid summary}
FRAMES=()
for a in $ARGS; do
  case $a in
    lab|bjorken|fluid) FRAMES+=("$a") ;;
    summary) ;;
    *) echo "unknown: $a (lab, bjorken, fluid, summary)"; exit 2 ;;
  esac
done
if ((${#FRAMES[@]})); then
  export -f one_frame log
  export OUT PROD PY EVENTS SEED OVERSAMPLE STEM
  printf '%s\n' "${FRAMES[@]}" | xargs -P "$JOBS" -I{} bash -c 'one_frame {}'
fi
case " $ARGS " in *" summary "*) summary ;; esac
log "done: $OUT"
