#!/usr/bin/env bash
# example/prod_AuAu_0_10_jet/run_jobs.sh
#
# Run N two-stage (jet) production jobs, one seed and one .h5 file per job, P at a time.
# This is ../prod_AuAu_0_10/run_jobs.sh driving run_prod_jet.py; the options (FIRST_SEED 0 =
# a campaign of unique seeds, --campaign), the skip-completed-jobs restart and Ctrl-C
# handling are the same.
#
#   ./run_jobs.sh [-j P] [--mps] [--stagger S] [--campaign NAME] NJOBS EVENTS_PER_JOB FIRST_SEED [OUTDIR] [run_prod_jet.py args...]
#   ./run_jobs.sh 20 25 0                              # unique seeds -> ./out/AuAu_0_10_jet_<start time>_00NN.h5
#   ./run_jobs.sh --campaign pth50 20 25 0             # named ./out/AuAu_0_10_jet_pth50_00NN.h5
#   ./run_jobs.sh 20 25 1                              # seeds 1..20 -> ./out/AuAu_0_10_jet_seed00NN.h5
#   ./run_jobs.sh -j 2 20 25 0 out_pgun --hard pgun --pgun-pt 40
#   ./run_jobs.sh 10 30 0 out_reuse3 --reuse 3         # one background per 3 jet events
#   ./run_jobs.sh -j 4 --mps 20 25 0                   # 4 at a time, GPU shared via CUDA MPS
#   ./run_jobs.sh -j 8 --mps --stagger 12 20 25 0     # the first 8 start 12 s apart (memory peaks spread)
#   ./run_jobs.sh 20 25 0 out_had --write-particlize both --particlize-only
#                                                      # only the _particlize.h5, no pair file
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROD_SCRIPT="$HERE/run_prod_jet.py"
export TAG_PREFIX="AuAu_0_10_jet_seed"
export PROD_OUTDIR="$HERE/out"
exec "$HERE/../prod_AuAu_0_10/run_jobs.sh" "$@"
