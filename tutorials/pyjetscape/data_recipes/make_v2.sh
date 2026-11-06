#!/usr/bin/env bash
# make_v2.sh SET [SET ...] -- how the tutorial data sets v2 were made (datasets.json).
#
#   OUT=~/JetScape/tutorial_data/v2 ./make_v2.sh pair_mini        # one set
#   ./make_v2.sh all                                              # every set, in order
#   ./make_v2.sh upload                                           # -> osdf:///fno4hic/tutorials/v2/
#
# Run with an environment that runs X-SCAPE (the Mac wheel, the production container, or a
# source build with the checkout's jetscape on PYTHONPATH).  The v2 sets were made on an
# Apple Silicon Mac (M3 Max), in the environment the xscape wheel 0.2.0 is built in, with
# MUSIC on the GPU (Metal).  Each set is one directory below $OUT; the seeds are fixed, so
# the same build gives the same events.  Uploaded sets are never overwritten: a changed set
# is a new version (v3, ...): the OSDF caches keep a replaced file for a long time.  (v1 was
# withdrawn before any release: its pair set's hadrons were remade with correlated sampling.)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
JS_CONTRIB=$(cd "$HERE/../../.." && pwd)
EX="$JS_CONTRIB/contribs/PyJetscape/example"
OUT=${OUT:-$HOME/JetScape/tutorial_data/v2}
PY=${PY:-python}
JOBS=${JOBS:-3}                            # parallel jobs (M3 Max: 3 x 5 threads)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-5} OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0
mkdir -p "$OUT"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── AuAu_pair_mini: one complete Au+Au background with 5 jets ─────────────────────────────
# 0-10% Au+Au at 200 GeV, PythiaGun pTHat 30-40 GeV with both hard partons at |y| < 0.6, one
# background reused for 5 jets (MUSIC_1 once, MUSIC_2 with each jet's deposition), both
# freeze-out surfaces stored; then stage 2: iSS on both legs with correlated sampling (200
# samples per jet event and for the background, sample k of each event paired with sample k
# of its background), 20 fragmentations per event; then the ROOT files.
# Tutorials 04, 06, 07 (and the derived wake summaries).
pair_mini() {
  local D="$OUT/AuAu_pair_mini"
  log "AuAu_pair_mini -> $D"
  (cd "$EX/prod_AuAu_0_10_jet" &&
   $PY run_prod_jet.py --events 5 --reuse 5 --seed 101 --pthat-min 30 --pthat-max 40 \
       --parton-ymax 0.6 --write-particlize both --outdir "$D" &&
   $PY hadronize.py "$D/AuAu_0_10_jet_seed0101_particlize.h5" \
       --correlated --common-seeds --oversample 200 --n-frag 20 &&
   $PY run_h5toROOT.py "$D" --writer uproot)
}

# ── AuAu_stats: hadron-level Au+Au events in three pTHat windows ──────────────────────────
# 4 jobs x 30 events: per background 2 jets in each of the windows 10-20, 20-30, 30-50 GeV
# (hard partons at |y| < 0.6), so 120 events (40 per window) on 20 backgrounds.  The
# hadrons are made in the job (--hadrons-only: no pair or particlize file): 10 oversamples
# per jet event, 20 per background, 10 fragmentations per event, no positions.  The
# windows' cross sections and event weights are recorded; the ROOT files and the campaign
# file follow.  Tutorials 05, 08, 09, 10.
stats() {
  local D="$OUT/AuAu_stats"
  log "AuAu_stats -> $D"
  (cd "$EX/prod_AuAu_0_10_jet" &&
   printf '%s\n' 201 202 203 204 | xargs -P "$JOBS" -I{} \
     $PY run_prod_jet.py --events 30 --seed {} --campaign stats \
       --pthat-bins 10-20,20-30,30-50 --jets-per-bin 2 --parton-ymax 0.6 \
       --hadrons-only --oversample 10 --n-frag 10 --no-x --outdir "$D" &&
   mkdir -p "$OUT/AuAu_stats_primaries" &&       # the undecayed hadrons (for an afterburner):
   mv "$D"/*_primary.h5 "$OUT/AuAu_stats_primaries/" &&   # not part of the set
   $PY run_h5toROOT.py "$D" --writer uproot -j "$JOBS")
}

# ── pp_ref: the vacuum reference of AuAu_stats ────────────────────────────────────────────
# The same PythiaGun windows and parton cut, Matter's shower in vacuum, Colorless
# hadronization: 3 jobs x 1500 events, 10 fragmentations per event; then the ROOT files.
# Tutorials 08, 10.
pp_ref() {
  local D="$OUT/pp_ref"
  log "pp_ref -> $D"
  (cd "$EX/prod_AuAu_0_10_jet" &&
   printf '%s\n' 401 402 403 | xargs -P "$JOBS" -I{} \
     $PY run_prod_vacuum.py --events 1500 --seed {} --campaign ppref \
       --pthat-bins 10-20,20-30,30-50 --parton-ymax 0.6 --n-frag 10 --no-x \
       --outdir "$D" &&
   $PY run_h5toROOT.py "$D" --writer uproot -j "$JOBS")
}

# ── OO_mini: 15 O+O events with initial-state radiation ───────────────────────────────────
# O+O at 200 GeV, b < 3 fm (about 0-24%), PythiaIsrGun pTHat > 20 GeV, iMATTER ISR, Matter +
# LBT in the medium: the hydro file, the particlize file, the jet hadrons and 50 bulk
# oversamples per event, made in the job; then the ROOT files.  Tutorials 02 (without
# X-SCAPE), 04, 05.
oo_mini() {
  local D="$OUT/OO_mini"
  log "OO_mini -> $D"
  (cd "$EX/prod_OO_isr" &&
   $PY run_prod_oo_isr.py --events 15 --seed 301 --bulk-hadrons 50 --outdir "$D" &&
   $PY ../prod_AuAu_0_10_jet/run_h5toROOT.py "$D" --writer uproot --tags bulk_bg,jet_frag)
}

# ── derived: campaign summaries the tutorials read instead of a whole campaign ────────────
# The wake summaries of AuAu_pair_mini (wake_observables.py, wake_hadrons.py), and the
# minimum-bias extra-hard-scattering histograms of the Pb+Pb campaign of
# analysis/runs/pbpb_extra_hard (extra_hard_hists.py; Bulk and flow on the website).
# Tutorials 07, 13.
derived() {
  local D="$OUT/derived"
  log "derived -> $D"
  mkdir -p "$D"
  (cd "$EX/analysis" &&
   $PY wake_observables.py "$OUT/AuAu_pair_mini" -j "$JOBS" -o "$D/wake_observables.h5" &&
   $PY wake_hadrons.py "$OUT/AuAu_pair_mini" -j "$JOBS" -o "$D/wake_hadrons.h5")
  cp "$EX/analysis/runs/pbpb_extra_hard/data/"{extra_hard_hists.npz,campaign_events.npz,campaign_bulk.json} "$D/"
}

# ── upload: every set to osdf:///fno4hic/tutorials/v2/ (needs a login: js_osdf.py login) ──
upload() {
  for d in AuAu_pair_mini AuAu_stats pp_ref OO_mini derived; do
    [ -d "$OUT/$d" ] || continue
    log "upload $d"
    python3 "$JS_CONTRIB/utils/remote_transfer/js_osdf.py" upload "$OUT/$d" --what all \
      --prefix tutorials/v2 -j 4
  done
}

case "${1:-}" in
  ""|-h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac
for s in "$@"; do
  case "$s" in
    pair_mini) pair_mini ;;
    stats) stats ;;
    pp_ref) pp_ref ;;
    oo_mini) oo_mini ;;
    derived) derived ;;
    upload) upload ;;
    all) pair_mini; stats; pp_ref; oo_mini; derived ;;
    *) echo "make_v2.sh: unknown set $s" >&2; exit 1 ;;
  esac
done
log "done: $*"
