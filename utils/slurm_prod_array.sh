#!/bin/bash
# utils/slurm_prod_array.sh -- stage 1 (GPU) campaigns on a SLURM cluster, in the production
# container (Apptainer/Singularity): prod_AuAu_0_10_jet (default) or prod_OO_isr (PROD=oo).
# See docs/README_2stage.md (section 1, per-GPU sizing; section 2, SLURM) and
# BuildContainerProd.md, "SLURM".
#
# One array task = one GPU, filled with P production jobs at once (run_jobs.sh -j P).  With
# the current image (cu124-final_optim) a job keeps ~1 core busy on average and the GPU only
# a little, so a GPU needs ~10 jobs, and the cores per GPU set the rate.  Separate SLURM jobs
# can't share a GPU unless the site offers --gres=shard/mps.
#
#   sbatch utils/slurm_prod_array.sh                    # the settings below: 10 GPUs
#   sbatch --array=0-19 utils/slurm_prod_array.sh       # 20 GPUs
#   CAMPAIGN=c2 EXTRA_ARGS="--pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 \
#     --write-particlize both" sbatch --array=0-19 utils/slurm_prod_array.sh
#   PROD=oo CAMPAIGN=OO_c1 EXTRA_ARGS="--bulk-hadrons 10 --no-particlize --no-hydro-file" \
#     sbatch --mem=40G utils/slurm_prod_array.sh        # O+O, hadron files only
#   CAMPAIGN=c3 EXTRA_ARGS="--pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 --parton-ymax 0.6 \
#     --hadrons-only" sbatch --array=0-19 utils/slurm_prod_array.sh   # Au+Au, hadron files only
#
# Per GPU (docs/README_2stage.md section 1; extrapolated from 2 x RTX 3090 and the GB10, none
# of these GPUs measured).  Give sbatch the resources and P the jobs; the script sets
# OMP_NUM_THREADS = cpus-per-task / P:
#
#   GPU    P (jobs)  --cpus-per-task  --mem Au+Au  --mem O+O   expected events/h per GPU
#   V100      8         12-16          64G          24G        Au+Au 700-950,    O+O 4,200-5,700
#   A40     10-12       16-24          90-100G      32G        Au+Au 1,100-1,400, O+O 7,000-8,500
#   A100    10-12       16-24          90-100G      32G        Au+Au 1,000-1,500, O+O 6,000-8,800
#   L40S    18-22       24-32          135-165G     48G        Au+Au 2,000-2,900, O+O 12,000-17,000
#
# Au+Au: ~7 GB host memory per job (5 GB with --particlize-only), plus ~8 GB per node and
# 10-15 % for the page cache SLURM counts.  O+O: ~1.5 GB per job.  At least 2 threads per job.
#
# Resubmit the same command to resume: each task skips its finished jobs.
#
# Output, per task t = SLURM_ARRAY_TASK_ID:
#   $WORK/$CAMPAIGN/t<ttt>/          the task's run_jobs.sh OUTDIR (campaign <CAMPAIGN>_t<ttt>)
#   $WORK/$CAMPAIGN/seeds_used.tsv   the seed registry, shared by all tasks
#   $WORK/lbt_cache/                 LBT's shared table cache (Au+Au)
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
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=xscape_prod_%A_%a.out

# ── Settings (environment variables override them) ────────────────────────────────────
SIF=${SIF:-$HOME/xscape_prod.sif}           # apptainer pull xscape_prod.sif docker://jhputschke/xscape-prod:cu124-final_optim
WORK=${WORK:-${SCRATCH:-$HOME}/xscape_prod} # bound to /work; writable, shared by the tasks
PROD=${PROD:-auau}                          # auau: prod_AuAu_0_10_jet, oo: prod_OO_isr
CAMPAIGN=${CAMPAIGN:-}                      # letters, digits, . _ - (default: per PROD below)
P=${P:-10}                                  # production jobs at once per GPU (table above)
NJOBS=${NJOBS:-40}                          # jobs (= output files) per array task
EVENTS=${EVENTS:-}                          # events per job (default: 45 Au+Au, a multiple
                                            # of 15 for the pT-hat-window campaigns; 60 O+O)
SEED_MODE=${SEED_MODE:-registry}            # registry | ranges
SEED_BASE=${SEED_BASE:-1}                   # ranges: the first seed of task 0
MUSIC_CUDA_SYNC=${MUSIC_CUDA_SYNC:-block}   # block: the host sleeps while it waits for the
                                            # GPU, ~30 % less CPU per event (auto: CUDA's default)
LBT_CACHE=${LBT_CACHE:-}                    # Au+Au: LBT's tables shared read-only by all jobs
                                            # of a node (-0.6 GB and 6-11 s per job); a path
                                            # inside the container, default
                                            # /work/lbt_cache/lbt_tables_v1.bin; none: off
STAGGER=${STAGGER:-}                        # s between the first P jobs' starts (memory peaks
                                            # apart); default 12 Au+Au, 0 O+O
USE_MPS=${USE_MPS:-0}                       # 1: run_jobs.sh --mps (+5-12 % where it works;
                                            # never tried on a cluster: test it first)
CHECK_DEPOSITION=${CHECK_DEPOSITION:-1}     # Au+Au: after the jobs, flag jobs whose droplets
                                            # per event are below half the task's median
BINDS=${BINDS:-}                            # more binds, e.g. "$HOME/xml:/xml"
# Arguments for run_prod_jet.py / run_prod_oo_isr.py, e.g. EXTRA_ARGS="--user-xml /xml/PbPb.xml"
EXTRA_ARGS=${EXTRA_ARGS:-}

EXAMPLE=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example

# ── Checks ───────────────────────────────────────────────────────────────────────────
set -u
die() { echo "slurm_prod_array.sh: $*" >&2; exit 1; }
case $PROD in
  auau) PROD_DIR=$EXAMPLE/prod_AuAu_0_10_jet; : "${CAMPAIGN:=AuAu_0_10_jet}"; : "${EVENTS:=45}"
        : "${STAGGER:=12}"; : "${LBT_CACHE:=/work/lbt_cache/lbt_tables_v1.bin}" ;;
  oo)   PROD_DIR=$EXAMPLE/prod_OO_isr; : "${CAMPAIGN:=OO_isr}"; : "${EVENTS:=60}"
        : "${STAGGER:=0}"; LBT_CACHE=none; CHECK_DEPOSITION=0 ;;
  *)    die "PROD must be auau or oo, got '$PROD'" ;;
esac
TASK=${SLURM_ARRAY_TASK_ID:-0}
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
CT=$(command -v apptainer || command -v singularity) || die "neither apptainer nor singularity found"
[ -f "$SIF" ] || die "SIF=$SIF not found (apptainer pull xscape_prod.sif docker://jhputschke/xscape-prod:cu124-final_optim)"
case $CAMPAIGN in .*|-*|*[!A-Za-z0-9._-]*|'') die "CAMPAIGN '$CAMPAIGN': use letters, digits, '.', '_', '-'" ;; esac
for v in P NJOBS EVENTS SEED_BASE; do
  case ${!v} in ''|*[!0-9]*|0) die "$v must be a positive integer, got '${!v}'" ;; esac
done
case $STAGGER in ''|*[!0-9]*) die "STAGGER must be whole seconds, got '$STAGGER'" ;; esac
case $MUSIC_CUDA_SYNC in block|yield|spin|auto) ;; *) die "MUSIC_CUDA_SYNC: block, yield, spin or auto" ;; esac
[ "$P" -le "$CPUS" ] || die "P=$P jobs but only $CPUS CPUs (--cpus-per-task)"
OMP=$(( CPUS / P ))
[ "$OMP" -ge 2 ] || echo "slurm_prod_array.sh: warning: $OMP thread(s) per job; at least 2" \
  "are recommended (string deposition and freeze-out search run in OpenMP bursts):" \
  "lower P or raise --cpus-per-task" >&2
mkdir -p "$WORK/$CAMPAIGN" || die "cannot create $WORK/$CAMPAIGN"

# LBT's table cache: a file inside the container.  Under /work its directory is created here;
# elsewhere (e.g. node-local scratch bound with BINDS) it must exist.  The first job writes the
# file (atomically), the others map it read-only.
lbt_env=
if [ "$LBT_CACHE" != none ] && [ -n "$LBT_CACHE" ]; then
  case $LBT_CACHE in
    /work/*) mkdir -p "$WORK/$(dirname "${LBT_CACHE#/work/}")" || die "cannot create the LBT cache directory" ;;
  esac
  lbt_env=",LBT_TABLE_CACHE=$LBT_CACHE"
fi

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
binds="$WORK:/work${BINDS:+,$BINDS}"
opts=(); [ "$USE_MPS" = 1 ] && opts+=(--mps)
[ "$STAGGER" -gt 0 ] && opts+=(--stagger "$STAGGER")
# The MPS daemon's sockets go to node-local /tmp (short path, per SLURM job).
export MPS_DIR=/tmp/xscape-mps.${SLURM_JOB_ID:-$$}.$TASK

echo "[$(date +%F\ %T)] task $TASK on $(hostname): $PROD, $P jobs at once, OMP_NUM_THREADS=$OMP," \
     "$NJOBS jobs x $EVENTS events, seeds: $SEED_MODE$([ "$FIRST_SEED" -gt 0 ] && echo " from $FIRST_SEED")"
echo "  image $SIF; OUTDIR $WORK/$CAMPAIGN/$TAG; GPU(s): ${CUDA_VISIBLE_DEVICES:-all visible}"
lbt_show=${lbt_env:+$LBT_CACHE}
echo "  MUSIC_CUDA_SYNC=$MUSIC_CUDA_SYNC, LBT cache: ${lbt_show:-off}," \
     "stagger ${STAGGER} s, MPS $([ "$USE_MPS" = 1 ] && echo on || echo off)"
echo "  run_prod options: ${EXTRA_ARGS:-(none)}"
command -v nvidia-smi > /dev/null && nvidia-smi -L | sed 's/^/  /'

# shellcheck disable=SC2086  # EXTRA_ARGS: split into the driver's arguments
"$CT" exec --nv \
  --env "OMP_NUM_THREADS=$OMP,OMP_WAIT_POLICY=passive,MPS_DIR=$MPS_DIR,MUSIC_CUDA_SYNC=$MUSIC_CUDA_SYNC$lbt_env" \
  --pwd "$PROD_DIR" --bind "$binds" "$SIF" \
  ./run_jobs.sh -j "$P" ${opts[@]+"${opts[@]}"} --campaign "${CAMPAIGN}_$TAG" \
    "$NJOBS" "$EVENTS" "$FIRST_SEED" "$OUTDIR" $EXTRA_ARGS
rc=$?
echo "[$(date +%F\ %T)] task $TASK: run_jobs.sh exited with $rc"

# Per-job deposition check (Au+Au): once in ~250 jobs a whole job's jets lost almost no
# energy, with no warning in its log (docs/REPORT_FINAL_OPTIM.md section 4).  Report only;
# utils/check_deposition.py does the same over a whole campaign.
if [ "$CHECK_DEPOSITION" = 1 ]; then
  "$CT" exec --bind "$binds" "$SIF" python3 - "$OUTDIR" <<'PY'
import glob, json, statistics, sys
rows = []
for p in glob.glob(f"{sys.argv[1]}/*.json"):
    try:
        j = json.load(open(p))
    except (OSError, ValueError):
        continue
    if isinstance(j, dict) and j.get("droplets_total") is not None and j.get("events_written"):
        rows.append((j["droplets_total"] / j["events_written"], p))
if len(rows) < 3:
    print(f"deposition check: {len(rows)} finished job(s), too few to compare")
    sys.exit(0)
med = statistics.median(r for r, _ in rows)
bad = [(r, p) for r, p in sorted(rows) if r < 0.5 * med]
print(f"deposition check: {len(rows)} jobs, median {med:.1f} droplets per event, "
      f"{len(bad)} below half of it")
for r, p in bad:
    print(f"  REDO {p}: {r:.1f} droplets per event (move its files away and resubmit)")
PY
fi
exit $rc
