#!/bin/bash
# utils/slurm_prod_array.sh -- prod_AuAu_0_10_jet campaigns on a SLURM cluster, in the
# production container (Apptainer/Singularity).  See BuildContainerProd.md, "SLURM".
#
# One array task = one GPU, filled with P production jobs at once (run_jobs.sh -j P):
# one job alone keeps a GPU only partly busy (the CPU stages dominate an event), and
# separate SLURM jobs can't share a GPU unless the site offers --gres=shard/mps.
#
#   sbatch utils/slurm_prod_array.sh                    # the settings below
#   CAMPAIGN=PbPb NJOBS=40 sbatch utils/slurm_prod_array.sh
#   sbatch --array=0-19 utils/slurm_prod_array.sh       # 20 GPUs instead of 10
#
# Resubmit the same command to resume: each task skips its finished jobs.  Choose
# NJOBS x EVENTS so a task fits its --time (prod_AuAu_0_10_jet: ~30-50 s per event and
# job, P jobs at once).
#
# Output, per task t = SLURM_ARRAY_TASK_ID:
#   $WORK/$CAMPAIGN/t<ttt>/   the task's run_jobs.sh OUTDIR (campaign <CAMPAIGN>_t<ttt>)
#   $WORK/$CAMPAIGN/seeds_used.tsv   the seed registry, shared by all tasks
#
# Seeds (SEED_MODE):
#   registry  every job draws a new seed (run_jobs.sh FIRST_SEED 0), checked against the
#             shared registry under a file lock.  That lock must hold across nodes, so
#             the script stops if $WORK's file system is mounted with node-local or no
#             locks (Lustre localflock/noflock, NFS nolock or local_lock).
#   ranges    task t runs seeds SEED_BASE + t*NJOBS ... + NJOBS-1: no lock needed, but
#             keep the ranges of different campaigns apart yourself.
#
#SBATCH --job-name=xscape_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=xscape_prod_%A_%a.out

# ── Settings (environment variables override them) ────────────────────────────────────
SIF=${SIF:-$HOME/xscape_prod.sif}           # apptainer pull xscape_prod.sif docker://...
WORK=${WORK:-${SCRATCH:-$HOME}/xscape_prod} # bound to /work; writable, shared by the tasks
CAMPAIGN=${CAMPAIGN:-AuAu_0_10_jet}         # letters, digits, . _ -
P=${P:-4}                                   # production jobs at once per GPU (~22 GB RAM each;
                                            # ~9 GB alone since the 2026-10 memory fixes, keep
                                            # 22 until a -j 4 campaign has been measured)
NJOBS=${NJOBS:-40}                          # jobs (= .h5 files) per array task
EVENTS=${EVENTS:-25}                        # events per job
SEED_MODE=${SEED_MODE:-registry}            # registry | ranges
SEED_BASE=${SEED_BASE:-1}                   # ranges: the first seed of task 0
USE_MPS=${USE_MPS:-0}                       # 1: run_jobs.sh --mps (test it on the site first)
BINDS=${BINDS:-}                            # more binds, e.g. "$HOME/xml:/xml"
# Arguments for run_prod_jet.py, e.g. EXTRA_ARGS="--user-xml /xml/PbPb_0_10.xml".
EXTRA_ARGS=${EXTRA_ARGS:-}

PROD=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet

# ── Checks ───────────────────────────────────────────────────────────────────────────
set -u
die() { echo "slurm_prod_array.sh: $*" >&2; exit 1; }
TASK=${SLURM_ARRAY_TASK_ID:-0}
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
CT=$(command -v apptainer || command -v singularity) || die "neither apptainer nor singularity found"
[ -f "$SIF" ] || die "SIF=$SIF not found (apptainer pull xscape_prod.sif docker://jhputschke/xscape-prod:<tag>)"
case $CAMPAIGN in .*|-*|*[!A-Za-z0-9._-]*|'') die "CAMPAIGN '$CAMPAIGN': use letters, digits, '.', '_', '-'" ;; esac
for v in P NJOBS EVENTS SEED_BASE; do
  case ${!v} in ''|*[!0-9]*|0) die "$v must be a positive integer, got '${!v}'" ;; esac
done
[ "$P" -le "$CPUS" ] || die "P=$P jobs but only $CPUS CPUs (--cpus-per-task)"
mkdir -p "$WORK/$CAMPAIGN" || die "cannot create $WORK/$CAMPAIGN"

case $SEED_MODE in
  registry)
    FIRST_SEED=0
    # The registry's flock must work across nodes.  Node-local or missing locks show in
    # the mount options; a lock that is refused outright, by a test lock.
    if command -v findmnt > /dev/null; then
      read -r fstype opts < <(findmnt -n -o FSTYPE,OPTIONS -T "$WORK/$CAMPAIGN")
      bad=
      case $fstype in
        lustre) case ,$opts, in *,localflock,*|*,noflock,*) bad=1 ;; esac ;;
        nfs*)   case ,$opts, in *,nolock,*|*,local_lock=flock,*|*,local_lock=all,*) bad=1 ;; esac ;;
      esac
      [ -z "$bad" ] || die "$WORK is $fstype mounted with '$opts': file locks don't hold" \
        "across nodes, so tasks could draw the same seed. Use SEED_MODE=ranges, or a WORK" \
        "on a file system with cluster-wide locks."
    fi
    # flock(1) takes the same flock(2) lock as run_prod.py.
    if command -v flock > /dev/null; then
      flock -x -w 60 "$WORK/$CAMPAIGN/seeds_used.tsv" true 2>/dev/null ||
        die "a file lock on $WORK/$CAMPAIGN/seeds_used.tsv fails; use SEED_MODE=ranges"
    fi ;;
  ranges)
    FIRST_SEED=$(( SEED_BASE + TASK * NJOBS )) ;;
  *) die "SEED_MODE must be registry or ranges, got '$SEED_MODE'" ;;
esac

# ── Run ──────────────────────────────────────────────────────────────────────────────
TAG=$(printf "t%03d" "$TASK")
OUTDIR=/work/$CAMPAIGN/$TAG
OMP=$(( CPUS / P ))
binds="$WORK:/work${BINDS:+,$BINDS}"
mps=(); [ "$USE_MPS" = 1 ] && mps=(--mps)
# The MPS daemon's sockets go to node-local /tmp (short path, per SLURM job).
export MPS_DIR=/tmp/xscape-mps.${SLURM_JOB_ID:-$$}.$TASK

echo "[$(date +%F\ %T)] task $TASK on $(hostname): $P jobs at once, OMP_NUM_THREADS=$OMP," \
     "$NJOBS jobs x $EVENTS events, seeds: $SEED_MODE$([ "$FIRST_SEED" -gt 0 ] && echo " from $FIRST_SEED")"
echo "  image $SIF; OUTDIR $WORK/$CAMPAIGN/$TAG; GPU(s): ${CUDA_VISIBLE_DEVICES:-all visible}"
command -v nvidia-smi > /dev/null && nvidia-smi -L | sed 's/^/  /'

# shellcheck disable=SC2086  # EXTRA_ARGS: split into run_prod_jet.py arguments
"$CT" exec --nv \
  --env "OMP_NUM_THREADS=$OMP,OMP_WAIT_POLICY=passive,MPS_DIR=$MPS_DIR" \
  --pwd "$PROD" --bind "$binds" "$SIF" \
  ./run_jobs.sh -j "$P" ${mps[@]+"${mps[@]}"} --campaign "${CAMPAIGN}_$TAG" \
    "$NJOBS" "$EVENTS" "$FIRST_SEED" "$OUTDIR" $EXTRA_ARGS
rc=$?
echo "[$(date +%F\ %T)] task $TASK: run_jobs.sh exited with $rc"
exit $rc
