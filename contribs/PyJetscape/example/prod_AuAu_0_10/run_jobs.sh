#!/usr/bin/env bash
# example/prod_AuAu_0_10/run_jobs.sh
#
# Run N production jobs on one GPU, one seed (and one .h5 file) per job, P at a time.
#
#   ./run_jobs.sh [-j P] [--mps] NJOBS EVENTS_PER_JOB FIRST_SEED [OUTDIR] [extra run_prod.py args...]
#   ./run_jobs.sh 20 25 1                         # seeds 1..20 -> ./out/AuAu_0_10_seed00NN.h5
#   ./run_jobs.sh -j 2 20 25 1                    # same, two jobs at a time
#   ./run_jobs.sh -j 4 --mps 20 25 1              # four at a time, sharing the GPU via CUDA MPS
#   ./run_jobs.sh 20 25 1 out_eta2p5 --grid grid_x10_eta2p5.yaml
#
# -j P (default 1) keeps P jobs running at once, with bit-identical output per seed.  Each
# job runs in its own working directory (see run_prod.py), so they can start together.
#
# --mps runs the jobs as clients of a CUDA MPS (Multi-Process Service) daemon started for
# this campaign, so their kernels share the GPU instead of being time-sliced; stopped again
# at the end, also on Ctrl-C.  Output is bit-identical.  prod_AuAu_0_10_jet on the GB10
# (events/h): -j 3 155 -> 163, -j 4 159 -> 179 with --mps; no gain for -j 1.  See
# ../prod_AuAu_0_10_jet/BENCHMARK_GB10.md.  The daemon's socket directory must have a short
# path (~100 characters for its UNIX sockets): $MPS_DIR, default
# ${XDG_RUNTIME_DIR:-/tmp}/xscape-mps.<pid>.
#
# macOS (Metal): --mps does not apply.  With -j > 1 split the cores between the jobs, or
# the OpenMP threads oversubscribe them and -j 3 gains nothing: prod_AuAu_0_10_jet on an
# M3 Max (16 cores), events/h: -j 1 132; -j 3 148 by default, 213 with
#   OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 ./run_jobs.sh -j 3 ...
# (OMP_NUM_THREADS ~ cores / P).  See ../prod_AuAu_0_10_jet/BENCHMARK_M3MAX.md.  The
# script runs under macOS's bash 3.2.
#
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
MPS=0
while [ $# -gt 0 ]; do
  case $1 in
    -j)    [ $# -ge 2 ] || usage; PAR=$2; shift 2 ;;
    -j*)   PAR=${1#-j}; shift ;;
    --mps) MPS=1; shift ;;
    *)     break ;;
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

# ---- CUDA MPS: one daemon for this campaign, its jobs as clients (--mps)
MPS_CREATED=0
mps_stop() {
  [ "$MPS" -eq 1 ] || return 0
  echo quit | nvidia-cuda-mps-control > /dev/null 2>&1
  local n; n=$(grep -c "NEW CLIENT" "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>/dev/null)
  echo "[$(date +%F\ %T)] MPS: daemon stopped (${n:-0} client connection(s))"
  if [ "$MPS_CREATED" -eq 1 ]; then          # only what this script made
    rm -rf "${MPS_DIR:?}/pipe" "${MPS_DIR:?}/log"; rmdir "$MPS_DIR" 2>/dev/null
  fi
  MPS=0
}
if [ "$MPS" -eq 1 ]; then
  command -v nvidia-cuda-mps-control > /dev/null ||
    { echo "--mps: nvidia-cuda-mps-control not found" >&2; exit 1; }
  MPS_DIR=${MPS_DIR:-${XDG_RUNTIME_DIR:-/tmp}/xscape-mps.$$}
  sock="$MPS_DIR/pipe/control"
  if [ ${#sock} -gt 100 ]; then
    echo "--mps: '$sock' is ${#sock} characters; MPS's UNIX sockets need ~100 or fewer" \
         "(it then fails silently). Set MPS_DIR to a shorter directory." >&2; exit 1
  fi
  [ -e "$MPS_DIR" ] || MPS_CREATED=1
  mkdir -p "$MPS_DIR/pipe" "$MPS_DIR/log"
  export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe" CUDA_MPS_LOG_DIRECTORY="$MPS_DIR/log"
  if ! nvidia-cuda-mps-control -d; then
    echo "--mps: could not start the MPS daemon in $MPS_DIR (another one running?):" >&2
    tail -5 "$MPS_DIR/log/control.log" >&2 2>/dev/null
    MPS=0; exit 1
  fi
  trap mps_stop EXIT
  echo "[$(date +%F\ %T)] MPS: daemon started, pipe directory $CUDA_MPS_PIPE_DIRECTORY"
fi

# Running jobs as two parallel indexed arrays (pids[i] runs seeds[i]) and a polling reap,
# not an associative array and `wait -n -p`: those need bash >= 5.1, and macOS ships 3.2.
# The ${a[@]+...} forms keep `set -u` quiet on empty arrays in bash < 4.4.
pids=(); seeds=()
# Background jobs of a non-interactive shell ignore Ctrl-C, so pass it on to them.
trap 'trap - INT TERM; echo "interrupted: stopping seeds ${seeds[*]+${seeds[*]}}" >&2;
      [ ${#pids[@]} -gt 0 ] && kill ${pids[@]+"${pids[@]}"} 2>/dev/null; wait; exit 130' INT TERM

failed=()
reap() {   # wait for one running job to finish and report it
  local i pid rc seed
  while :; do                  # bash reaps finished children itself, so kill -0 fails
    for i in ${pids[@]+"${!pids[@]}"}; do
      pid=${pids[$i]}
      kill -0 "$pid" 2>/dev/null && continue
      wait "$pid"; rc=$?      # the saved exit status
      seed=${seeds[$i]}; unset "pids[$i]" "seeds[$i]"
      break 2
    done
    sleep 1
  done
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
  while [ ${#pids[@]} -ge "$PAR" ]; do reap; done
  echo "[$(date +%F\ %T)] seed $seed: job $((k + 1))/$NJOBS, $EVENTS events -> $OUTDIR/$tag.h5"
  python "$PROD_SCRIPT" --events "$EVENTS" --seed "$seed" --outdir "$OUTDIR" ${1+"$@"} \
    > "$OUTDIR/$tag.log" 2>&1 &
  pids+=("$!"); seeds+=("$seed")
done
while [ ${#pids[@]} -gt 0 ]; do reap; done
trap - INT TERM
mps_stop

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
