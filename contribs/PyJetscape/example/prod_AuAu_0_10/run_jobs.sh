#!/usr/bin/env bash
# example/prod_AuAu_0_10/run_jobs.sh
#
# Run N production jobs on one GPU, one seed (and one .h5 file) per job, P at a time.
#
#   ./run_jobs.sh [-j P] NJOBS EVENTS_PER_JOB FIRST_SEED [OUTDIR] [extra run_prod.py args...]
#   ./run_jobs.sh 20 25 1                         # seeds 1..20 -> ./out/AuAu_0_10_seed00NN.h5
#   ./run_jobs.sh -j 2 20 25 1                    # same, two jobs at a time (~1.7x throughput)
#   ./run_jobs.sh 10 50 101 /data/AuAu_0_10       # seeds 101..110
#   ./run_jobs.sh 20 25 1 out_eta2p5 --grid grid_x10_eta2p5.yaml
#
# -j P (default 1) keeps P jobs running at once.  On the GB10, -j 2 gives ~1.66x the
# throughput of -j 1 (each job's CPU stages overlap the other's GPU evolution), with
# bit-identical output per seed; each job holds ~3-5 GB of host RAM.
# A failed job is logged and the others continue; re-run just that seed later.
# Jobs whose .json summary already says complete are skipped, so an interrupted campaign can
# be restarted with the same command.  Give each grid YAML its own OUTDIR: the skip test
# looks at the seed and event count only.
#
# Other productions reuse this script through three environment variables (defaults: this
# folder's): PROD_SCRIPT (the per-job driver), TAG_PREFIX (file names <prefix><seed>.*) and
# PROD_OUTDIR (the default OUTDIR); see ../prod_AuAu_0_10_jet/run_jobs.sh.
set -u

usage() { sed -n '6,11p' "$0"; exit 2; }

PAR=1
while [ $# -gt 0 ]; do
  case $1 in
    -j)  [ $# -ge 2 ] || usage; PAR=$2; shift 2 ;;
    -j*) PAR=${1#-j}; shift ;;
    *)   break ;;
  esac
done
case $PAR in ''|*[!0-9]*|0) echo "-j needs a positive integer, got '$PAR'" >&2; exit 2 ;; esac

[ $# -ge 3 ] || usage
NJOBS=$1; EVENTS=$2; SEED0=$3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_SCRIPT=${PROD_SCRIPT:-"$HERE/run_prod.py"}
TAG_PREFIX=${TAG_PREFIX:-AuAu_0_10_seed}
shift 3
OUTDIR="${PROD_OUTDIR:-$HERE/out}"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then   # an optional OUTDIR before the options
  OUTDIR=$1; shift
fi
OUTDIR="$(mkdir -p "$OUTDIR" && cd "$OUTDIR" && pwd)"

# Pythia needs PYTHIA8DATA only where its compiled-in xmldoc path is invalid (a relocated
# conda Pythia, e.g. js_fno); there importing pyjetscape_core aborts. Homebrew's is fine.
if [ -z "${PYTHIA8DATA:-}" ] && ! python -c "import sys; sys.path.insert(0, '$HERE/../../python'); import jetscape" >/dev/null 2>&1; then
  echo "PYTHIA8DATA is not set and importing pyjetscape_core fails without it:" \
       "point it to Pythia's xmldoc ('conda activate js_fno' sets it)." >&2; exit 1
fi

# Background jobs of a non-interactive shell ignore Ctrl-C, so pass it on to them.
declare -A running=()   # pid -> seed
trap 'trap - INT TERM; echo "interrupted: stopping seeds ${running[*]}" >&2;
      [ ${#running[@]} -gt 0 ] && kill "${!running[@]}" 2>/dev/null; wait; exit 130' INT TERM

failed=()
reap() {   # wait for one running job to finish and report it
  local pid rc
  wait -n -p pid "${!running[@]}"; rc=$?
  local seed=${running[$pid]}; unset "running[$pid]"
  local tag; tag=$(printf "%s%04d" "$TAG_PREFIX" "$seed")
  if [ "$rc" -eq 0 ]; then
    echo "[$(date +%F\ %T)] seed $seed: ok"
  else
    echo "[$(date +%F\ %T)] seed $seed: FAILED (see $OUTDIR/$tag.log)"; failed+=("$seed")
  fi
}

for (( k = 0; k < NJOBS; k++ )); do
  seed=$(( SEED0 + k ))
  tag=$(printf "%s%04d" "$TAG_PREFIX" "$seed")
  if [ -f "$OUTDIR/$tag.json" ] && \
     python -c "import json,sys; d=json.load(open('$OUTDIR/$tag.json')); \
                sys.exit(d['events_written'] != $EVENTS)" 2>/dev/null; then
    echo "[$(date +%F\ %T)] seed $seed: already complete, skipping"; continue
  fi
  while [ ${#running[@]} -ge "$PAR" ]; do reap; done
  echo "[$(date +%F\ %T)] seed $seed: job $((k + 1))/$NJOBS, $EVENTS events -> $OUTDIR/$tag.h5"
  python "$PROD_SCRIPT" --events "$EVENTS" --seed "$seed" --outdir "$OUTDIR" "$@" \
    > "$OUTDIR/$tag.log" 2>&1 &
  running[$!]=$seed
done
while [ ${#running[@]} -gt 0 ]; do reap; done
trap - INT TERM

# Events longer than tau.max_ntau keep only their first max_ntau frames.  That is the point
# of setting it (e.g. early times only), so this is a count, not an error.
ncut=$(python - "$OUTDIR" "$TAG_PREFIX" <<'EOF'
import glob, json, os, sys
print(sum(d.get("events_cut_at_max_ntau", 0) + d.get("legs_cut_at_max_ntau", 0)
          for d in (json.load(open(p))
                    for p in glob.glob(os.path.join(sys.argv[1], sys.argv[2] + "*.json")))))
EOF
)
if [ "${ncut:-0}" -gt 0 ]; then
  echo "note: $ncut event(s) ran past tau.max_ntau and kept only its first frames" \
       "(set max_ntau: 0 to keep whole events)."
fi
if [ ${#failed[@]} -gt 0 ]; then
  echo "failed seeds: ${failed[*]}"; exit 1
fi
echo "all $NJOBS jobs done: $OUTDIR"
