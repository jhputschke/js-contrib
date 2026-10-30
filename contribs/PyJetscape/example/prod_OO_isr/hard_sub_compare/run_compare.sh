#!/usr/bin/env bash
# example/prod_OO_isr/hard_sub_compare/run_compare.sh
#
# Two O+O ISR campaigns that differ only in the hard-momentum subtraction:
#   sub/     the hard partons' momentum is taken from the colliding nucleons (default)
#   nosub/   --no-subtraction (the reference: the energy is counted twice)
# Both: 0-80% centrality in the geometry, b < 5.6 fm (3dMCGlauber draws b uniformly in b^2
# and redraws until Npart >= 2; inside X-SCAPE cenMin/cenMax do nothing).  5.6 fm is the
# 80th percentile of b of inelastic O+O at 200 GeV with this XML's 3dMCGlauber settings
# (100k standalone 3dMCGlauber events, sigma_inel = 1.55 b; b_max 12 gives 0-100%).  No
# tighter cut, so a centrality bias of the subtraction can be measured rather than cut
# away; experiments do not measure the most peripheral 20% either.  pTHat windows 10-20, 20-30, 30-40 GeV, both outgoing hard partons at
# |y| < 0.6, hadrons only (bulk with 10 iSS oversamples, jet hadrons from iColored), no
# positions.  The hard scattering sits at one binary collision drawn uniformly: weight the
# events with Ncoll for the hard-process rate (the notebook does).
# The same seeds in both: the same Pythia hard events, event by event; the 3dMCGlauber
# geometry differs from the second event of a job on (compare ensembles).
#
#   ./run_compare.sh                       # 40 jobs x 75 events each = 1000 per window
#   ./run_compare.sh > run_compare.log 2>&1 &
# Resumable: re-run the same command after an interruption (complete jobs are skipped).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NJOBS=${NJOBS:-40}            # jobs per campaign
EVENTS=${EVENTS:-75}          # events per job: a multiple of the 3 windows
SEED0=${SEED0:-101}           # seeds SEED0 .. SEED0+NJOBS-1, in both campaigns
P=${P:-8}                     # jobs at a time (GB10: ~2,200 events/h at -j 8 with MPS)
B_MIN=${B_MIN:-0}             # impact parameter range [fm]: 0-5.6 is 0-80%,
B_MAX=${B_MAX:-5.6}           # 0-12 every inelastic collision
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

# --matter-only: these campaigns ran with Matter alone (the default before 2026-10-21);
# jobs added to them must too.  Matter + LBT is a new comparison in new directories.
common=(--b-min "$B_MIN" --b-max "$B_MAX"
        --pthat-bins 10-20,20-30,30-40 --parton-ymax 0.6 --parton-y-mode both
        --bulk-hadrons 10 --no-particlize --no-hydro-file --no-x --matter-only)

# The two campaigns alternate in blocks of BLOCK jobs (run_jobs.sh skips complete jobs, so
# growing NJOBS continues where the last block stopped): an interrupted run leaves about
# equal statistics in both.
BLOCK=${BLOCK:-16}
n=0
while [ "$n" -lt "$NJOBS" ]; do
  n=$(( n + BLOCK < NJOBS ? n + BLOCK : NJOBS ))
  "$HERE/../run_jobs.sh" -j "$P" --mps "$n" "$EVENTS" "$SEED0" "$HERE/sub" "${common[@]}"
  "$HERE/../run_jobs.sh" -j "$P" --mps "$n" "$EVENTS" "$SEED0" "$HERE/nosub" "${common[@]}" \
    --no-subtraction
done
