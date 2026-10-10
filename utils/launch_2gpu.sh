#!/usr/bin/env bash
# launch_2gpu.sh: prod_AuAu_0_10_jet on both GPUs of this machine (2 x RTX 3090, 24 cores),
# with the best measured setup (README_launch.md): one container per GPU, each running
# run_jobs.sh -j 8 --mps with OMP_NUM_THREADS=4  (~520-535 events/h, ~87 GB host memory).
#
#   ./launch_2gpu.sh [upload options] NJOBS EVENTS_PER_JOB [FIRST_SEED] [OUTBASE] [run_prod_jet.py args...]
#   ./launch_2gpu.sh 64 25                        # 64 jobs x 25 events, unique seeds
#   ./launch_2gpu.sh 64 25 0 /work/campA --write-particlize both
#   ./launch_2gpu.sh 16 3 1 /work/test            # seeds 1..16 (GPU0 1..8, GPU1 9..16)
#   ./launch_2gpu.sh 64 25 0 /data/campB          # output on the second disk (/data)
#   ./launch_2gpu.sh 64 45 0 /work/camp_pth3 --pthat-bins 10-20,20-30,30-40 --jets-per-bin 5 \
#       --parton-ymax 0.6 --write-particlize both
#                    # suggested campaign: 3 pTHat windows x 5 jets = one background per 15 jet
#                    # events (EVENTS must be a multiple of 15), |y| < 0.6 on the leading parton.
#                    # Measured: ~810 events/h (vs 450 without bins), 97 GB host memory, ~0.25 GB
#                    # disk per event (README_launch.md)
#   IMAGE=jhputschke/xscape-prod:cu124-cuda-sync-mem-test JOBS_PER_GPU=10 OMP_THREADS=3 \
#   MUSIC_CUDA_SYNC=block ./launch_2gpu.sh --lbt-cache /work/lbt_cache/lbt_tables_v1.bin \
#       --stagger 12 60 45 0 /data/camp_pth3 --campaign pth3 --pthat-bins 10-20,20-30,30-40 \
#       --jets-per-bin 5 --parton-ymax 0.6 --write-particlize both
#                    # the same campaign at 10 jobs/GPU: shared LBT tables, staggered start,
#                    # blocking CUDA sync. Measured: ~805 events/h by makespan (+8 % over 8/GPU),
#                    # ~105 GB host memory peak (docs/BENCHMARK_2x3090.md, 2026-10-10). Needs an image
#                    # with LBT_TABLE_CACHE and run_jobs.sh --stagger (js-contrib 593fcc4 or later).
#                    # NJOBS a multiple of 20: 60 x 45 = 2,700 events, ~3 h, ~0.6 TB
#   ./launch_2gpu.sh --upload AuAu_pth3_c1 --delete pair 64 45 0 /data/c1 ...
#                    # + upload finished jobs to osdf:///fno4hic/AuAu_pth3_c1/gpuN while running,
#                    # deleting the verified pair files locally
#
# Upload options (before NJOBS; see upload_follow.py, README_launch.md):
#   --upload REMOTE      upload each finished job to osdf:///fno4hic/REMOTE/gpu0|gpu1 while the
#                        campaign runs (host side, upload_follow.py; log OUTBASE/upload.log).
#                        Needs ./remote_transfer/js_osdf.py login once.
#   --delete MODE        none (default) | pair (<stem>.h5) | all (+ <stem>_particlize.h5):
#                        delete those local files once their upload is verified
#   --keep-free SIZE     with pair/all: delete only while OUTBASE's disk has less free (e.g. 200G)
#   --verify-every N     download every Nth file before deleting it and compare (default 20; 0 off)
#
# Memory options (before NJOBS; what lets 10 jobs/GPU with particlize fit, see above):
#   --lbt-cache FILE     share LBT's 0.66 GB of tables read-only between all jobs and skip their
#                        parsing (LBT_TABLE_CACHE). FILE is a host path (or /work/...); if it does
#                        not exist, one 1-event job on GPU0 makes it first (~1 min, log FILE.init.log).
#                        One file per LBT-tables version; a file that doesn't match is ignored by
#                        the jobs (they parse the tables as before).
#   --stagger S          each GPU starts its first JOBS_PER_GPU jobs S seconds apart, so their
#                        memory peaks don't coincide (12 for 8-10 jobs/GPU). Per-event times then
#                        look shorter than they are; use the makespan for events/h.
#
# Disk guard (before NJOBS):
#   --min-free SIZE      stop the campaign (both containers) when OUTBASE's disk has less free than
#                        SIZE, checked every minute; also refuse to start below it (default 50G;
#                        0 off).  The jobs running then are lost; free space and run the same
#                        command again to resume.
#
# NJOBS is the total; GPU0 gets the first half (rounded up), GPU1 the rest.  Output goes to
# OUTBASE/gpu0 and OUTBASE/gpu1.  OUTBASE (default /work/out) is either /work/... ($WORKDIR,
# mounted as /work) or any other host directory, e.g. /data/campB, which is created and mounted
# into the containers at the same path.  FIRST_SEED 0 (default): every job draws a unique seed,
# recorded in OUTBASE/seeds_used.tsv (shared by both GPUs, file-locked).  Re-running the same
# command resumes: completed jobs are skipped.  Ctrl-C stops both containers.
#
# Environment overrides: JOBS_PER_GPU (8), OMP_THREADS (4), IMAGE, WORKDIR ($HOME/prod_test).
# MUSIC_CUDA_SYNC (block|yield|spin|auto) is passed to the jobs if set; block: ~30 % less CPU
# per event, needed for 10 jobs/GPU to pay off.
# Why one container per GPU: two MPS daemons in one container hang the second GPU's jobs.
# Options, campaign default and how to run a campaign: README_launch.md.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() { awk 'NR > 1 && /^set -u/ { exit } NR > 1' "$0"; exit 2; }
REMOTE=; upargs=(); MINFREE=50G; LBTCACHE=; STAGGER=
while [ $# -gt 0 ]; do
  case $1 in
    --upload)       [ $# -ge 2 ] || usage; REMOTE=$2; shift 2 ;;
    --delete|--keep-free|--verify-every)
                    [ $# -ge 2 ] || usage; upargs+=("$1" "$2"); shift 2 ;;
    --min-free)     [ $# -ge 2 ] || usage; MINFREE=$2; shift 2 ;;
    --lbt-cache)    [ $# -ge 2 ] || usage; LBTCACHE=$2; shift 2 ;;
    --stagger)      [ $# -ge 2 ] || usage; STAGGER=$2; shift 2 ;;
    -h|--help)      usage ;;
    --*)            echo "unknown option $1 (launcher options go before NJOBS)" >&2; usage ;;
    *)              break ;;
  esac
done
[ ${#upargs[@]} -eq 0 ] || [ -n "$REMOTE" ] || { echo "--delete/--keep-free/--verify-every need --upload" >&2; exit 2; }
case $STAGGER in ''|[0-9]|[0-9][0-9]|[0-9][0-9][0-9]) ;;
  *) echo "--stagger needs whole seconds, got '$STAGGER'" >&2; exit 2 ;; esac
[ $# -ge 2 ] || usage
NJOBS=$1; EVENTS=$2; SEED0=${3:-0}; OUTBASE=${4:-/work/out}
shift $(( $# < 4 ? $# : 4 ))
J=${JOBS_PER_GPU:-8}; OMP=${OMP_THREADS:-4}
IMAGE=${IMAGE:-jhputschke/xscape-prod:cu124}
WORKDIR=${WORKDIR:-$HOME/prod_test}
mounts=(-v "$WORKDIR:/work")
case $OUTBASE in
  /work|/work/*) ;;                              # inside WORKDIR, already mounted
  *) OUTBASE=$(realpath -m "$OUTBASE")           # a host directory: mount it at the same path
     case $OUTBASE in /|/opt|/opt/*|/usr|/usr/*|/bin*|/lib*|/etc*|/tmp|/proc*|/sys*|/dev*)
       echo "OUTBASE $OUTBASE would hide a directory of the image; choose e.g. /data/..." >&2
       exit 2 ;; esac
     mkdir -p "$OUTBASE" && [ -w "$OUTBASE" ] ||
       { echo "OUTBASE $OUTBASE: cannot create or write it" >&2; exit 1; }
     mounts+=(-v "$OUTBASE:$OUTBASE") ;;
esac
HOST_OUTBASE=$OUTBASE
LBT_H=; LBT_C=                                   # the LBT cache file on the host / in the containers
if [ -n "$LBTCACHE" ]; then
  WD_ABS=$(realpath -m "$WORKDIR")
  case $LBTCACHE in
    /work|/work/*) LBT_C=$LBTCACHE; LBT_H=$WD_ABS${LBTCACHE#/work} ;;
    *) LBT_H=$(realpath -m "$LBTCACHE")
       case $LBT_H in
         "$WD_ABS"/*) LBT_C=/work${LBT_H#"$WD_ABS"} ;;
         /|/opt|/opt/*|/usr|/usr/*|/bin*|/lib*|/etc*|/tmp|/tmp/*|/proc*|/sys*|/dev*)
           echo "--lbt-cache $LBT_H: choose a directory under WORKDIR or e.g. /data/..." >&2; exit 2 ;;
         *) LBT_C=$LBT_H                         # mount its directory at the same path
            d=$(dirname "$LBT_H")
            case " ${mounts[*]} " in *" $d:$d "*) ;; *) mounts+=(-v "$d:$d") ;; esac ;;
       esac ;;
  esac
  mkdir -p "$(dirname "$LBT_H")" && [ -w "$(dirname "$LBT_H")" ] ||
    { echo "--lbt-cache: cannot create or write $(dirname "$LBT_H")" >&2; exit 1; }
fi
case $OUTBASE in /work|/work/*) HOST_OUTBASE=$WORKDIR${OUTBASE#/work}; mkdir -p "$HOST_OUTBASE" ;; esac

to_bytes() { numfmt --from=si "$(echo "${1%[bB]}" | tr a-z A-Z)" 2>/dev/null; }
free_bytes() { df -B1 --output=avail "$HOST_OUTBASE" | tail -1 | tr -d ' '; }
MINFREE_B=$(to_bytes "$MINFREE") || { echo "--min-free: not a size: $MINFREE (e.g. 50G, 1T, 0)" >&2; exit 2; }
if [ "$MINFREE_B" -gt 0 ] && [ "$(free_bytes)" -lt "$MINFREE_B" ]; then
  echo "only $(numfmt --to=si "$(free_bytes)") free on $HOST_OUTBASE, below --min-free $MINFREE: not starting" >&2
  exit 1
fi
rm -f "$HOST_OUTBASE/.stopped_low_disk"

UPID=; GPID=
if [ -n "$REMOTE" ]; then              # check the upload can work before starting the campaign
  "$HERE/remote_transfer/js_osdf.py" status > /dev/null 2>&1 ||
    { echo "--upload: no OSDF login: $HERE/remote_transfer/js_osdf.py login" >&2; exit 1; }
  "$HERE/upload_follow.py" --help > /dev/null 2>&1 ||
    { echo "--upload: $HERE/upload_follow.py does not run" >&2; exit 1; }
  rm -f "$HOST_OUTBASE/.upload_final"
fi

if [ -n "$STAGGER" ] && ! docker run --rm --entrypoint grep "$IMAGE" -q -e '--stagger)' \
       ../prod_AuAu_0_10/run_jobs.sh; then
  echo "--stagger: run_jobs.sh in $IMAGE has no --stagger (needs js-contrib 64e39ba or later)" >&2
  exit 1
fi
if [ -n "$LBT_H" ] && [ ! -s "$LBT_H" ]; then  # make the cache once, before the jobs need it
  echo "LBT table cache $LBT_H not there yet: making it with one 1-event job on GPU0 (~1 min)"
  docker run --rm --name "xscape_lbtcache_$$" --gpus '"device=0"' --user "$(id -u):$(id -g)" \
      -e OMP_NUM_THREADS="$OMP" -e LBT_TABLE_CACHE="$LBT_C" "${mounts[@]}" "$IMAGE" \
      python run_prod_jet.py --events 1 --seed 1 --outdir /tmp/lbt_cache_init \
      --seed-registry none > "$LBT_H.init.log" 2>&1
  if ! grep -q "LBT: wrote the table cache" "$LBT_H.init.log" || [ ! -s "$LBT_H" ]; then
    echo "--lbt-cache: the cache was not written (an image without LBT_TABLE_CACHE?);" \
         "see $LBT_H.init.log" >&2
    exit 1
  fi
  echo "LBT table cache written: $LBT_H ($(stat -c %s "$LBT_H") bytes)"
fi
jobenv=(); [ -n "$LBT_C" ] && jobenv+=(-e "LBT_TABLE_CACHE=$LBT_C")
rjopts=(); [ -n "$STAGGER" ] && rjopts+=(--stagger "$STAGGER")

N0=$(( (NJOBS + 1) / 2 )); N1=$(( NJOBS - N0 ))
cids=()
stop() {
  echo "stopping containers" >&2
  [ -n "$GPID" ] && kill "$GPID" 2>/dev/null
  docker stop -t 30 "${cids[@]}" > /dev/null 2>&1; docker rm "${cids[@]}" > /dev/null 2>&1
  [ -n "$UPID" ] && { kill -TERM "$UPID" 2>/dev/null; wait "$UPID"; }
  exit 130
}
trap stop INT TERM
for g in 0 1; do
  n=$(( g == 0 ? N0 : N1 )); [ "$n" -gt 0 ] || continue
  s=$(( SEED0 == 0 ? 0 : SEED0 + g * N0 ))
  cids+=("$(docker run -d --name "xscape_gpu${g}_$$" --gpus "\"device=$g\"" \
      --user "$(id -u):$(id -g)" -e OMP_NUM_THREADS="$OMP" -e OMP_WAIT_POLICY=passive \
      -e MUSIC_CUDA_SYNC ${jobenv[@]+"${jobenv[@]}"} "${mounts[@]}" "$IMAGE" \
      ./run_jobs.sh -j "$J" --mps ${rjopts[@]+"${rjopts[@]}"} "$n" "$EVENTS" "$s" "$OUTBASE/gpu$g" "$@")") || exit 1
  echo "GPU$g: $n jobs x $EVENTS events, container ${cids[-1]:0:12}," \
       "log: docker logs -f xscape_gpu${g}_$$"
done
if [ "$MINFREE_B" -gt 0 ]; then          # the disk guard
  (
    while sleep "${GUARD_INTERVAL:-60}"; do
      a=$(free_bytes)
      if [ "$a" -lt "$MINFREE_B" ]; then
        echo "[$(date '+%F %T')] LOW DISK: $(numfmt --to=si "$a") free on $HOST_OUTBASE, below" \
             "--min-free $MINFREE: stopping the campaign. Free space, then run the same command" \
             "to resume." | tee "$HOST_OUTBASE/.stopped_low_disk"
        docker stop -t 30 "${cids[@]}" > /dev/null 2>&1
        exit 0
      fi
    done
  ) &
  GPID=$!
fi
if [ -n "$REMOTE" ]; then
  "$HERE/upload_follow.py" "$HOST_OUTBASE" "$REMOTE" --follow --follow-pid $$ \
      ${upargs[@]+"${upargs[@]}"} >> "$HOST_OUTBASE/upload.log" 2>&1 &
  UPID=$!
  echo "upload: -> osdf:///fno4hic/$REMOTE/gpuN ${upargs[*]:-}, log: $HOST_OUTBASE/upload.log"
fi
rc=0
for c in "${cids[@]}"; do
  r=$(docker wait "$c"); [ "$r" -eq 0 ] || rc=1
  docker logs "$c" 2>&1 | tail -2
  docker rm "$c" > /dev/null
done
[ -n "$GPID" ] && kill "$GPID" 2>/dev/null
if [ -e "$HOST_OUTBASE/.stopped_low_disk" ]; then
  rc=1; echo "STOPPED for low disk space: $(cat "$HOST_OUTBASE/.stopped_low_disk")" >&2
fi
if [ -n "$UPID" ]; then                 # the uploader's last pass, then its summary
  touch "$HOST_OUTBASE/.upload_final"
  wait "$UPID" || rc=1
  grep "pass\|WARNING\|ERROR\|FAILED" "$HOST_OUTBASE/upload.log" | tail -2
fi
exit $rc
