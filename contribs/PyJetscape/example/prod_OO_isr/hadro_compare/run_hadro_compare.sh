#!/usr/bin/env bash
# example/prod_OO_isr/hadro_compare/run_hadro_compare.sh
#
# O+O 200 GeV, Matter only, jets hadronized with iColoredHadronization: five campaigns with
# the same seeds, so the same Pythia hard events, the same geometry and the same Pythia
# seed per event (paired, event by event):
#   before/     the code before X-SCAPE lbt_colored_hadro (duplicate colour tags, no checks):
#               an X-SCAPE contrib 41bdf8c5 + js-contrib main 03d14fa tree and build, $BASELINE
#   after/      the same partons, the colour fixes (one tag counter per event):  --matter-only
#   reconnect/  Matter with recoils (recoil_on 1, the holes open colour lines), closure
#               reconnect (the default)
#   thermal/    the same partons as reconnect/, closure thermal
#   fake/       the same partons as reconnect/, closure fake
# Without recoils Matter opens no colour line and the three closures are identical, hence
# recoil_on 1 for them; no LBT anywhere.  Jet hadrons only (no iSS), with the Pythia input
# (--store-hadronization-input).  b < 5.6 fm (0-80%), pTHat windows 10-20, 20-30, 30-40
# GeV, both hard partons at |y| < 0.6, as hard_sub_compare/.
#
#   BASELINE=/path/to/X-SCAPE-baseline ./run_hadro_compare.sh > run_hadro_compare.log 2>&1 &
#   python compare_hadro.py                        # -> figures/, numbers for SUMMARY.md
# NJOBS, EVENTS, SEED0, P, BLOCK, BASELINE, BASE_BUILD, OUTROOT and OMP_NUM_THREADS
# (environment) change the defaults.
# Resumable: re-run the same command after an interruption (complete jobs are skipped).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NJOBS=${NJOBS:-40}            # jobs per campaign
EVENTS=${EVENTS:-75}          # events per job: a multiple of the 3 windows
SEED0=${SEED0:-101}           # seeds SEED0 .. SEED0+NJOBS-1, in every campaign
P=${P:-8}                     # jobs at a time
BLOCK=${BLOCK:-8}             # the campaigns alternate in blocks of BLOCK jobs
BASELINE=${BASELINE:-$HOME/JetScape/X-SCAPE-regbase}
BASE_BUILD=${BASE_BUILD:-$BASELINE/build_gpu_base}
OUTROOT=${OUTROOT:-$HERE}     # where before/ after/ reconnect/ thermal/ fake/ go
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

BASE_JOBS="$BASELINE/external_packages/js-contrib/contribs/PyJetscape/example/prod_OO_isr/run_jobs.sh"
for f in "$BASE_JOBS" "$BASE_BUILD/music_input"; do
  [ -e "$f" ] || { echo "run_hadro_compare.sh: $f missing (BASELINE, BASE_BUILD)" >&2; exit 1; }
done
if git -C "$BASELINE" merge-base --is-ancestor 4f644df7 HEAD 2>/dev/null; then
  echo "run_hadro_compare.sh: $BASELINE has the colour fixes; it must be the code before" >&2
  exit 1
fi

# Matter with recoils, no LBT: the O+O template without <Lbt> (recoil_on 1 stays)
mkdir -p "$HERE/xml"
python3 - "$HERE/../../config/OO_MCGlauber_MUSIC_isr.xml" "$HERE/xml/OO_matter_recoil.xml" <<'PY'
import sys, xml.etree.ElementTree as ET
tree = ET.parse(sys.argv[1]); eloss = tree.getroot().find("Eloss")
if eloss.find("Lbt") is not None:
    eloss.remove(eloss.find("Lbt"))
assert eloss.find("Matter/recoil_on").text.strip() == "1"
tree.write(sys.argv[2])
PY

common=(--b-min 0 --b-max 5.6 --pthat-bins 10-20,20-30,30-40 --parton-ymax 0.6
        --parton-y-mode both --no-particlize --no-hydro-file --no-x
        --store-hadronization-input)
n=0
while [ "$n" -lt "$NJOBS" ]; do
  n=$(( n + BLOCK < NJOBS ? n + BLOCK : NJOBS ))
  "$BASE_JOBS" -j "$P" --mps "$n" "$EVENTS" "$SEED0" "$OUTROOT/before" "${common[@]}" \
    --build "$BASE_BUILD"
  "$HERE/../run_jobs.sh" -j "$P" --mps "$n" "$EVENTS" "$SEED0" "$OUTROOT/after" \
    "${common[@]}" --matter-only
  for mode in reconnect thermal fake; do
    "$HERE/../run_jobs.sh" -j "$P" --mps "$n" "$EVENTS" "$SEED0" "$OUTROOT/$mode" \
      "${common[@]}" --user-xml "$HERE/xml/OO_matter_recoil.xml" --colour-closure "$mode"
  done
done
