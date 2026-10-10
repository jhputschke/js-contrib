#!/usr/bin/env bash
# example/prod_AuAu_0_10/run_jobs.sh
#
# Run N production jobs on one GPU, one seed (and one .h5 file) per job, P at a time.
#
#   ./run_jobs.sh [-j P] [--mps] [--stagger S] [--campaign NAME] NJOBS EVENTS_PER_JOB FIRST_SEED [OUTDIR] [run_prod.py args...]
#   ./run_jobs.sh 20 25 0                         # unique seeds -> ./out/AuAu_0_10_<start time>_00NN.h5
#   ./run_jobs.sh 20 25 0 --campaign mb_a         # the same, named ./out/AuAu_0_10_mb_a_00NN.h5
#   ./run_jobs.sh 20 25 1                         # seeds 1..20 -> ./out/AuAu_0_10_seed00NN.h5
#   ./run_jobs.sh -j 4 --mps 20 25 0              # four at a time, sharing the GPU via CUDA MPS
#   ./run_jobs.sh 20 25 0 out_eta2p5 --grid grid_x10_eta2p5.yaml
#
# FIRST_SEED 0 (a campaign): every job draws its own seed from OS entropy (run_prod.py
# --seed 0: 1..900000000, not in the seed registry OUTDIR/../seeds_used.tsv, recorded there,
# in the file and in its .json), and the files are numbered NNNN = 1..NJOBS under the campaign
# name: --campaign (before or after the numbers), else the start time YYYYMMDD-HHMMSS.  The name is kept in
# OUTDIR/run_jobs.campaign, so re-running the command resumes the same campaign (one campaign
# per OUTDIR).  FIRST_SEED > 0: seeds FIRST_SEED.. as given, files named by seed,
# <prefix><seed> or with --campaign <base>_<campaign>_seed<seed>: the same seeds give the
# same collisions, which is what paired
# comparisons of settings want and what campaigns meant as more statistics must avoid.
#
# -j P (default 1) keeps P jobs running at once, with bit-identical output per seed.  Each
# job runs in its own working directory (see run_prod.py), so they can start together.
#
# --mps runs the jobs as clients of a CUDA MPS (Multi-Process Service) daemon started for
# this campaign, so their kernels share the GPU instead of being time-sliced; stopped again
# at the end, also on Ctrl-C.  Output is bit-identical.  prod_AuAu_0_10_jet on the GB10
# (events/h): -j 3 155 -> 163, -j 4 159 -> 179 with --mps; no gain for -j 1.  See
# ../../../../docs/BENCHMARK_GB10.md.  The daemon's socket directory must have a short
# path (~100 characters for its UNIX sockets): $MPS_DIR, default
# ${XDG_RUNTIME_DIR:-/tmp}/xscape-mps.<pid>.
#
# macOS (Metal): --mps does not apply.  With -j > 1 split the cores between the jobs, or
# the OpenMP threads oversubscribe them and -j 3 gains nothing: prod_AuAu_0_10_jet on an
# M3 Max (16 cores), events/h: -j 1 132; -j 3 148 by default, 213 with
#   OMP_NUM_THREADS=5 OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0 ./run_jobs.sh -j 3 ...
# (OMP_NUM_THREADS ~ cores / P).  See ../../../../docs/BENCHMARK_M3MAX.md.  The
# script runs under macOS's bash 3.2.
#
# A failed job is logged and the others continue; re-run just that seed later.
# Jobs whose .json summary already says complete are skipped (with --write-particlize the
# particlize file must be complete too), so an interrupted campaign can
# be restarted with the same command.  Give each grid YAML its own OUTDIR: the skip test
# looks at the seed and event count only.
#
# Other productions reuse this script through three environment variables (defaults: this
# folder's): PROD_SCRIPT (the per-job driver), TAG_PREFIX (file names <prefix><seed>.*, and
# <prefix without _seed>_<campaign>_<NNNN>.* for a campaign) and PROD_OUTDIR (the default
# OUTDIR); see ../prod_AuAu_0_10_jet/run_jobs.sh.
set -u

usage() { sed -n '6,12p' "$0"; exit 2; }

PAR=1
MPS=0
STAGGER=0
CAMPAIGN=
while [ $# -gt 0 ]; do
  case $1 in
    -j)    [ $# -ge 2 ] || usage; PAR=$2; shift 2 ;;
    -j*)   PAR=${1#-j}; shift ;;
    --mps) MPS=1; shift ;;
    --stagger) [ $# -ge 2 ] || usage; STAGGER=$2; shift 2 ;;
    --campaign)   [ $# -ge 2 ] || usage; CAMPAIGN=$2; [ -n "$CAMPAIGN" ] || usage; shift 2 ;;
    --campaign=*) CAMPAIGN=${1#*=}; [ -n "$CAMPAIGN" ] || usage; shift ;;
    *)     break ;;
  esac
done
case $PAR in ''|*[!0-9]*|0) echo "-j needs a positive integer, got '$PAR'" >&2; exit 2 ;; esac
case $STAGGER in ''|*[!0-9]*) echo "--stagger needs whole seconds, got '$STAGGER'" >&2; exit 2 ;; esac

[ $# -ge 3 ] || usage
NJOBS=$1; EVENTS=$2; SEED0=$3
for v in "NJOBS=$NJOBS" "EVENTS_PER_JOB=$EVENTS" "FIRST_SEED=$SEED0"; do
  case ${v#*=} in ''|*[!0-9]*) echo "$v: needs a non-negative integer" >&2; usage ;; esac
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_SCRIPT=${PROD_SCRIPT:-"$HERE/run_prod.py"}
TAG_PREFIX=${TAG_PREFIX:-AuAu_0_10_seed}
NAME_BASE=${TAG_PREFIX%_seed}                  # AuAu_0_10_seed -> AuAu_0_10
shift 3
# The arguments after the numbers go to every job, except --campaign, which is this script's
# wherever it stands (it numbers the jobs; taken out first, so OUTDIR may follow it), and
# the options it sets per job itself.
set_campaign() {
  if [ -z "$1" ] || { [ -n "$CAMPAIGN" ] && [ "$CAMPAIGN" != "$1" ]; }; then
    echo "--campaign: given as '${CAMPAIGN}' and '$1'" >&2; exit 2
  fi
  CAMPAIGN=$1
}
[ -n "$CAMPAIGN" ] && set_campaign "$CAMPAIGN"
pass=()
while [ $# -gt 0 ]; do
  case $1 in
    --campaign)   [ $# -ge 2 ] || usage; set_campaign "$2"; shift 2 ;;
    --campaign=*) set_campaign "${1#*=}"; shift ;;
    --outdir|--outdir=*)
      echo "--outdir: give OUTDIR as the argument after FIRST_SEED instead, e.g." \
           "run_jobs.sh $NJOBS $EVENTS $SEED0 /work/out" >&2; exit 2 ;;
    --seed|--seed=*|--index|--index=*|--events|--events=*|--out|--out=*)
      echo "${1%%=*}: set by run_jobs.sh for every job (NJOBS EVENTS_PER_JOB FIRST_SEED" \
           "[OUTDIR]); don't pass it" >&2; exit 2 ;;
    *) pass+=("$1"); shift ;;
  esac
done
set -- ${pass[@]+"${pass[@]}"}
OUTDIR="${PROD_OUTDIR:-$HERE/out}"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then   # an optional OUTDIR before the options
  OUTDIR=$1; shift
fi
OUTDIR="$(mkdir -p "$OUTDIR" && cd "$OUTDIR" && pwd)" ||
  { echo "OUTDIR: cannot create it" >&2; exit 1; }

# Every job records its seed in the registry (run_prod.py --seed-registry; default
# seeds_used.tsv next to OUTDIR), so check once that it can be written rather than let every
# job fail.  The usual cause is a container: only the bound directories are writable, and
# Apptainer mounts the home directory itself, not its parent, so OUTDIR ~ puts the registry
# into the read-only image.
registry="$(dirname "$OUTDIR")/seeds_used.tsv"
for (( i = 1; i <= $#; i++ )); do
  case ${!i} in
    --seed-registry)   j=$(( i + 1 )); [ $j -le $# ] && registry=${!j} ;;
    --seed-registry=*) registry=${!i#*=} ;;
  esac
done
case $registry in
  [Nn][Oo][Nn][Ee]) ;;
  *) d=$(dirname "$registry")                   # run_prod.py creates missing directories
     while [ ! -e "$d" ]; do d=$(dirname "$d"); done
     if { [ -e "$registry" ] && [ ! -w "$registry" ]; } ||
        { [ ! -e "$registry" ] && [ ! -w "$d" ]; }; then
       echo "The seed registry $registry can't be written (read-only, or no permission; in a" \
            "container, is it inside a bound directory?). Put OUTDIR one level deeper, e.g." \
            "$OUTDIR/out, choose the file with --seed-registry PATH, or turn it off with" \
            "--seed-registry none (explicit seeds only: seed 0 then has no protection" \
            "against repeats)." >&2; exit 1
     fi ;;
esac


# ---- campaign: FIRST_SEED 0 (unique seeds) or --campaign names the files <base>_<name>_NNNN
if [ "$SEED0" -eq 0 ] || [ -n "$CAMPAIGN" ]; then
  stored=
  [ -f "$OUTDIR/run_jobs.campaign" ] && stored=$(cat "$OUTDIR/run_jobs.campaign")
  if [ -z "$CAMPAIGN" ]; then
    CAMPAIGN=${stored:-$(date +%Y%m%d-%H%M%S)}
  elif [ -n "$stored" ] && [ "$stored" != "$CAMPAIGN" ]; then
    echo "$OUTDIR holds campaign '$stored' (run_jobs.campaign), not '$CAMPAIGN': one campaign" \
         "per OUTDIR, so a re-run resumes it. Use another OUTDIR." >&2; exit 2
  fi
  case $CAMPAIGN in
    .*|-*|*[!A-Za-z0-9._-]*)
      echo "--campaign '$CAMPAIGN': use letters, digits, '.', '_', '-'" >&2; exit 2 ;;
  esac
  echo "$CAMPAIGN" > "$OUTDIR/run_jobs.campaign"
fi
job_tag() {    # file stem of job k (0-based): by job number when the seed is drawn, else by seed
  if [ "$SEED0" -eq 0 ]; then printf "%s_%s_%04d" "$NAME_BASE" "$CAMPAIGN" $(( $1 + 1 ))
  elif [ -n "$CAMPAIGN" ]; then printf "%s_%s_seed%04d" "$NAME_BASE" "$CAMPAIGN" $(( SEED0 + $1 ))
  else printf "%s%04d" "$TAG_PREFIX" $(( SEED0 + $1 )); fi
}
job_label() {  # how the log names job k
  if [ "$SEED0" -eq 0 ]; then printf "job %04d" $(( $1 + 1 ))
  else printf "seed %d" $(( SEED0 + $1 )); fi
}
# Written when the campaign ends (not on Ctrl-C), so ../prod_AuAu_0_10_jet/run_hadronize.py
# --follow knows no more files are coming; a new campaign in this OUTDIR removes it.
rm -f "$OUTDIR/run_jobs.finished"

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

# Running jobs as parallel indexed arrays (pids[i] runs jobs[i]) and a polling reap, not an
# associative array and `wait -n -p`: those need bash >= 5.1, and macOS ships 3.2.
# The ${a[@]+...} forms keep `set -u` quiet on empty arrays in bash < 4.4.
pids=(); jobs_k=()
# Background jobs of a non-interactive shell ignore Ctrl-C, so pass it on to them.
trap 'trap - INT TERM; echo "interrupted: stopping ${#pids[@]} running job(s)" >&2;
      [ ${#pids[@]} -gt 0 ] && kill ${pids[@]+"${pids[@]}"} 2>/dev/null; wait; exit 130' INT TERM

failed=()
reap() {   # wait for one running job to finish and report it
  local i pid rc k
  while :; do                  # bash reaps finished children itself, so kill -0 fails
    for i in ${pids[@]+"${!pids[@]}"}; do
      pid=${pids[$i]}
      kill -0 "$pid" 2>/dev/null && continue
      wait "$pid"; rc=$?      # the saved exit status
      k=${jobs_k[$i]}; unset "pids[$i]" "jobs_k[$i]"
      break 2
    done
    sleep 1
  done
  local tag label; tag=$(job_tag "$k"); label=$(job_label "$k")
  if [ "$rc" -eq 0 ]; then
    echo "[$(date +%F\ %T)] $label: ok"
  else
    echo "[$(date +%F\ %T)] $label: FAILED (see $OUTDIR/$tag.log)"; failed+=("$label")
  fi
}

for (( k = 0; k < NJOBS; k++ )); do
  seed=$(( SEED0 == 0 ? 0 : SEED0 + k ))
  tag=$(job_tag "$k"); label=$(job_label "$k")
  naming=()
  [ -n "$CAMPAIGN" ] && naming=(--campaign "$CAMPAIGN" --index $(( k + 1 )))
  if [ -f "$OUTDIR/$tag.json" ] && \
     python -c "import json,sys; d=json.load(open('$OUTDIR/$tag.json')); \
                sys.exit(d['events_written'] != $EVENTS or \
                         d.get('particlize_events_written', $EVENTS) != $EVENTS)" \
       2>/dev/null; then
    echo "[$(date +%F\ %T)] $label: already complete, skipping"; continue
  fi
  while [ ${#pids[@]} -ge "$PAR" ]; do reap; done
  # --stagger S: the first PAR jobs start S seconds apart, so their memory peaks (at
  # the end of each event) do not coincide; later jobs start as slots free up anyway.
  if [ "$STAGGER" -gt 0 ] && [ "${nstarted:-0}" -gt 0 ] && [ "${nstarted:-0}" -lt "$PAR" ]; then
    sleep "$STAGGER"
  fi
  nstarted=$(( ${nstarted:-0} + 1 ))
  echo "[$(date +%F\ %T)] $label: job $((k + 1))/$NJOBS, $EVENTS events -> $OUTDIR/$tag.h5"
  python "$PROD_SCRIPT" --events "$EVENTS" --seed "$seed" ${naming[@]+"${naming[@]}"} \
    --outdir "$OUTDIR" ${1+"$@"} > "$OUTDIR/$tag.log" 2>&1 &
  pids+=("$!"); jobs_k+=("$k")
done
while [ ${#pids[@]} -gt 0 ]; do reap; done
trap - INT TERM
mps_stop
if [ "$SEED0" -eq 0 ]; then
  seeds_desc="seeds from OS entropy (each job's .json)"
else
  seeds_desc="seeds $SEED0..$(( SEED0 + NJOBS - 1 ))"
fi
echo "finished $(date +%F\ %T): ${CAMPAIGN:+campaign $CAMPAIGN, jobs 1..$NJOBS, }$seeds_desc," \
     "$EVENTS events each" > "$OUTDIR/run_jobs.finished"
if [ ${#failed[@]} -gt 0 ]; then
  echo "failed: ${failed[*]}" >> "$OUTDIR/run_jobs.finished"
fi

# Events longer than tau.max_ntau keep only their first max_ntau frames.  That is the point
# of setting it (e.g. early times only), so this is a count, not an error.
ncut=$(python - "$OUTDIR" "$(job_tag 0 | sed 's/[0-9]*$//')" <<'EOF'
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
  echo "failed: ${failed[*]}"; exit 1
fi
echo "all $NJOBS jobs done: $OUTDIR"
