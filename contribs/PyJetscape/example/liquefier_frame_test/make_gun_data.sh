#!/usr/bin/env bash
# make_gun_data.sh -- the single-parton frame test's data: one quark per event, shot from the
# origin at fixed (pT, y) points into one LHC background, the same events in the three frames
# of the CausalLiquefier (lab, bjorken, fluid), each with its hydro pair, particlize file and
# wake hadrons; then the summary compare_frames_gun.ipynb reads.
#
#   ./make_gun_data.sh                       # all three frames, then the summary
#   ./make_gun_data.sh fluid                 # one frame (the summary needs all three)
#   ./make_gun_data.sh summary               # only the summary (gun_summary.py)
#   ./make_gun_data.sh clean                 # delete the h5 files, keep summary.npz and logs
#
# The events: 0-30% Pb+Pb at 5.02 TeV (config/PbPb_0_30_ZJet.xml, its background, energy loss
# and liquefier; the Z+jet campaign's settings), one background (SEED) reused for every event.
# X-SCAPE's PGun (<points>, branch liquefier-rest-frame) shoots one parton (PID, default 2: a
# u quark, as the Z's recoil mostly is) from x = y = z = t = 0 at phi = 0 with pT and
# rapidity from the grid GRID ("pT,...:y,...", default 20,50,100:0,1,2,3,4: 15 points),
# PER events per point (default 5): 75 events.  Event i uses point i mod 15, so all points
# share the background.  The deposits' frame is the only difference between the three runs:
# backgrounds, partons and droplets are bit-identical (gun_summary.py checks it).
#
# Hadrons: the wake only (bulk jet - background, correlated sampling with common seeds,
# OVERSAMPLE samples per event and for the background); no fragmentation of the lone quark.
# About 40 min per frame on an M3 Max (a Pb+Pb job needs ~20 GB, so JOBS=2), ~35 GB per frame
# before `clean`.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PROD="$HERE/../prod_AuAu_0_10_jet"
CONFIG="$HERE/../config"
OUT=${OUT:-$HOME/JetScape/liquefier_gun_test}
GRID=${GRID:-20,50,100:0,1,2,3,4}
PER=${PER:-5}
PID=${PID:-2}
SEED=${SEED:-102}
OVERSAMPLE=${OVERSAMPLE:-100}
JOBS=${JOBS:-2}
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7} OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0

npt=$(echo "${GRID%%:*}" | tr ',' '\n' | grep -c .)
ny=$(echo "${GRID##*:}" | tr ',' '\n' | grep -c .)
EVENTS=$((npt * ny * PER))
STEM=$(printf "PbPb_0_30_gun_seed%04d" "$SEED")

log() { echo "[$(date +%H:%M:%S)] $*"; }

one_frame() {
  local fr=$1 D="$OUT/$1"
  mkdir -p "$D"
  log "frame $fr -> $D ($EVENTS events: $GRID x $PER, parID $PID)"
  (cd "$PROD" &&
   $PY run_prod_jet.py --hard pgun --pgun-grid "$GRID" --pgun-per-point "$PER" \
       --pgun-id "$PID" --events "$EVENTS" --seed "$SEED" --out "$STEM.h5" \
       --user-xml "$CONFIG/PbPb_0_30_ZJet.xml" --grid "$CONFIG/grid_PbPb.yaml" \
       --seed-registry none --write-particlize both --liquefier-frame "$fr" \
       --outdir "$D" > "$D/prod.log" 2>&1 &&
   $PY hadronize.py "$D/${STEM}_particlize.h5" --tags bulk_jet,bulk_bg \
       --correlated --common-seeds --oversample "$OVERSAMPLE" \
       > "$D/hadronize.log" 2>&1)
  log "frame $fr done ($(du -sh "$D" | cut -f1))"
}

summary() {
  log "summary -> $OUT/summary.npz"
  $PY "$HERE/gun_summary.py" "$OUT" -o "$OUT/summary.npz"
}

clean() {
  [[ -f "$OUT/summary.npz" ]] || { echo "no $OUT/summary.npz: run the summary first"; exit 2; }
  find "$OUT"/{lab,bjorken,fluid} -maxdepth 1 -name '*.h5' -delete
  log "cleaned: $(du -sh "$OUT" | cut -f1) left"
}

ARGS=${*:-lab bjorken fluid summary}
FRAMES=()
for a in $ARGS; do
  case $a in
    lab|bjorken|fluid) FRAMES+=("$a") ;;
    summary|clean) ;;
    *) echo "unknown: $a (lab, bjorken, fluid, summary, clean)"; exit 2 ;;
  esac
done
if ((${#FRAMES[@]})); then
  export -f one_frame log
  export OUT PROD CONFIG PY GRID PER PID SEED OVERSAMPLE EVENTS STEM
  printf '%s\n' "${FRAMES[@]}" | xargs -P "$JOBS" -I{} bash -c 'one_frame {}'
fi
case " $ARGS " in *" summary "*) summary ;; esac
case " $ARGS " in *" clean "*) clean ;; esac
log "done: $OUT"
