#!/usr/bin/env bash
# example/prod_AuAu_0_10_jet/run_jobs.sh
#
# Run N two-stage (jet) production jobs, one seed (and one AuAu_0_10_jet_seedNNNN.h5) per
# job, P at a time.  This is ../prod_AuAu_0_10/run_jobs.sh driving run_prod_jet.py; the
# options, the skip-completed-seeds restart and Ctrl-C handling are the same.
#
#   ./run_jobs.sh [-j P] [--mps] NJOBS EVENTS_PER_JOB FIRST_SEED [OUTDIR] [extra run_prod_jet.py args...]
#   ./run_jobs.sh 20 25 1                              # seeds 1..20 -> ./out/
#   ./run_jobs.sh -j 2 20 25 1 out_pgun --hard pgun --pgun-pt 40
#   ./run_jobs.sh 10 30 1 out_reuse3 --reuse 3         # one background per 3 jet events
#   ./run_jobs.sh -j 4 --mps 20 25 1                   # 4 at a time, GPU shared via CUDA MPS
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROD_SCRIPT="$HERE/run_prod_jet.py"
export TAG_PREFIX="AuAu_0_10_jet_seed"
export PROD_OUTDIR="$HERE/out"
exec "$HERE/../prod_AuAu_0_10/run_jobs.sh" "$@"
