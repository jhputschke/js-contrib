#!/usr/bin/env bash
# example/prod_OO_isr/run_jobs.sh
#
# Run N O+O ISR production jobs, one seed per job, P at a time.  This is
# ../prod_AuAu_0_10/run_jobs.sh driving run_prod_oo_isr.py; the options (FIRST_SEED 0 = a
# campaign of unique seeds, --campaign), the skip-completed-jobs restart and Ctrl-C
# handling are the same.
#
#   ./run_jobs.sh [-j P] [--mps] [--stagger S] [--campaign NAME] NJOBS EVENTS_PER_JOB FIRST_SEED [OUTDIR] [run_prod_oo_isr.py args...]
#   ./run_jobs.sh 20 25 0                              # unique seeds -> ./out/OO_isr_<start time>_00NN.*
#   ./run_jobs.sh 20 25 1                              # seeds 1..20 -> ./out/OO_isr_seed00NN.*
#   ./run_jobs.sh -j 4 --mps 20 25 0 out_b5 --b-max 5  # 4 at a time, GPU shared via CUDA MPS
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROD_SCRIPT="$HERE/run_prod_oo_isr.py"
export TAG_PREFIX="${TAG_PREFIX:-OO_isr_seed}"   # or the caller's (TAG_PREFIX=..._seed)
export PROD_OUTDIR="$HERE/out"
exec "$HERE/../prod_AuAu_0_10/run_jobs.sh" "$@"
