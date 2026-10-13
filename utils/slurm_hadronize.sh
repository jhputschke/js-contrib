#!/bin/bash
# utils/slurm_hadronize.sh -- stage 2 (CPU) on a SLURM cluster: run_hadronize.py on the
# particlize files of a stage-1 campaign, in the same container, without a GPU.  See
# docs/README_2stage.md, section 3.
#
# One array task = one directory of the campaign: task t hadronizes $WORK/$CAMPAIGN/t<ttt>/,
# the OUTDIR of stage-1 task t (slurm_prod_array.sh), P files at once.  So the --array of
# stage 1 covers the campaign; a task whose directory doesn't exist does nothing.
# INPUTS="..." (directories, particlize files or quoted globs, as paths inside the container,
# /work/...) hadronizes those instead.
#
#   CAMPAIGN=c2 sbatch --array=0-19 utils/slurm_hadronize.sh
#   CAMPAIGN=c2 HAD_ARGS="--oversample 400 --oversample-bg per-pthat-bin --n-frag 50 \
#     --keep-bits-p 12 --keep-bits-x 8" sbatch --array=0-19 utils/slurm_hadronize.sh
#   PROD=oo CAMPAIGN=OO_c1 sbatch --array=0-9 utils/slurm_hadronize.sh      # O+O: the bulk
#   CAMPAIGN=c2 INPUTS="/work/c2/t*/*_particlize.h5" sbatch --array=0 utils/slurm_hadronize.sh
#
# hadronize.py uses one core per process: P = cpus-per-task processes, OMP_NUM_THREADS=1.
# Memory ~1.6 GB per process (~1.7 GB with 2000-sample backgrounds): --mem ~2 GB x P.
# Time: Au+Au, both legs at 500 oversamples, ~10.6 s per event and process (8.8 s with
# --correlated); one process does ~340 events/h, so ~6-7 processes keep up with a stage 1 of
# 2,000 events/h.  Resubmit the same command to finish an interrupted pass.
#
# Choose HAD_ARGS once per campaign and keep them for every task and resubmission: complete
# files are skipped whatever settings made them.
#
#SBATCH --job-name=xscape_had
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=xscape_had_%A_%a.out

# ── Settings (environment variables override them) ────────────────────────────────────
SIF=${SIF:-$HOME/xscape_prod.sif}           # the stage-1 image; no GPU needed
WORK=${WORK:-${SCRATCH:-$HOME}/xscape_prod} # bound to /work, as in slurm_prod_array.sh
PROD=${PROD:-auau}                          # auau: all three tags; oo: the bulk (bulk_bg)
CAMPAIGN=${CAMPAIGN:-}                      # default: AuAu_0_10_jet / OO_isr, as in stage 1
THREADS=${THREADS:-1}                       # OMP_NUM_THREADS per hadronize.py process
P=${P:-}                                    # processes at once (default cpus-per-task / THREADS)
INPUTS=${INPUTS:-}                          # instead of the task's directory
# Options for hadronize.py (via run_hadronize.py); defaults per PROD below.  For pT-hat-window
# campaigns analysed per window add --oversample-bg per-pthat-bin; for jet - background with
# fewer oversamples, --correlated (not together with --oversample-bg).
HAD_ARGS=${HAD_ARGS:-}
BINDS=${BINDS:-}                            # more binds, e.g. "$HOME/xml:/xml" (--hadronize-xml)

EXAMPLE=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example

# ── Checks ───────────────────────────────────────────────────────────────────────────
set -u
die() { echo "slurm_hadronize.sh: $*" >&2; exit 1; }
case $PROD in
  auau) : "${CAMPAIGN:=AuAu_0_10_jet}"
        : "${HAD_ARGS:=--oversample 400 --n-frag 50 --keep-bits-p 12 --keep-bits-x 8}" ;;
  oo)   : "${CAMPAIGN:=OO_isr}"
        : "${HAD_ARGS:=--oversample 100}"
        # the jet hadrons of O+O are made in the job (iColored); hadronize.py makes the bulk
        case " $HAD_ARGS " in *" --tags"*) ;; *) HAD_ARGS="--tags bulk_bg $HAD_ARGS" ;; esac ;;
  *)    die "PROD must be auau or oo, got '$PROD'" ;;
esac
TASK=${SLURM_ARRAY_TASK_ID:-0}
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
CT=$(command -v apptainer || command -v singularity) || die "neither apptainer nor singularity found"
[ -f "$SIF" ] || die "SIF=$SIF not found"
case $CAMPAIGN in .*|-*|*[!A-Za-z0-9._-]*|'') die "CAMPAIGN '$CAMPAIGN': use letters, digits, '.', '_', '-'" ;; esac
case $THREADS in ''|*[!0-9]*|0) die "THREADS must be a positive integer" ;; esac
: "${P:=$(( CPUS / THREADS ))}"
case $P in ''|*[!0-9]*|0) die "P must be a positive integer, got '$P'" ;; esac

if [ -n "$INPUTS" ]; then
  inputs=$INPUTS
  where=$INPUTS
else
  TAG=$(printf "t%03d" "$TASK")
  if [ ! -d "$WORK/$CAMPAIGN/$TAG" ]; then
    echo "slurm_hadronize.sh: task $TASK: no $WORK/$CAMPAIGN/$TAG, nothing to do"
    exit 0
  fi
  inputs=/work/$CAMPAIGN/$TAG
  where=$WORK/$CAMPAIGN/$TAG
fi
binds="$WORK:/work${BINDS:+,$BINDS}"

# ── Run ──────────────────────────────────────────────────────────────────────────────
echo "[$(date +%F\ %T)] task $TASK on $(hostname): $PROD, $P processes x $THREADS thread(s)," \
     "inputs $where"
echo "  hadronize options: $HAD_ARGS"

set -f   # a glob in INPUTS is for run_hadronize.py inside the container, not for this shell
# shellcheck disable=SC2086  # inputs (globs expanded by run_hadronize.py) and HAD_ARGS
"$CT" exec --env "OMP_NUM_THREADS=$THREADS,OMP_WAIT_POLICY=passive,OPENBLAS_NUM_THREADS=1" \
  --pwd "$EXAMPLE/prod_AuAu_0_10_jet" --bind "$binds" "$SIF" \
  ./run_hadronize.py $inputs -j "$P" $HAD_ARGS
rc=$?
echo "[$(date +%F\ %T)] task $TASK: run_hadronize.py exited with $rc"
exit $rc
