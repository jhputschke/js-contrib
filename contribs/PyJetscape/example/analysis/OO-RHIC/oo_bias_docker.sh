#!/usr/bin/env bash
# analysis/OO-RHIC/oo_bias_docker.sh: oo_bias.py on a multi-GPU Docker host, one container per
# GPU, each running its shard of the job list (--shard g/N), with the settings benchmarked on
# the 2 x RTX 3090 box (docs/BENCHMARK_2x3090.md: 8 jobs per GPU, OMP 3, MPS,
# MUSIC_CUDA_SYNC=block, OMP_WAIT_POLICY=passive; the host is the limit, more jobs per GPU do
# not help).
#
#   ./oo_bias_docker.sh [--dev] OUTROOT SAMPLES... [oo_bias.py options]
#   ./oo_bias_docker.sh /data/OO-RHIC_bulk screen --list            # what would run, no GPU
#   ./oo_bias_docker.sh /data/OO-RHIC_bulk screen --limit 8         # timing test: 4 jobs per GPU
#   ./oo_bias_docker.sh /data/OO-RHIC_bulk screen                   # the screening, 10k per sample
#   ./oo_bias_docker.sh /data/OO-RHIC_bulk screen --events 5000     # 5k per sample
#
# OUTROOT holds one directory per sample (oo_bias.py --out) and is mounted at the same path.
# The same command again resumes (finished jobs are skipped).  Containers run detached:
#   docker logs -f oo_bias_gpu0          docker stop oo_bias_gpu0 oo_bias_gpu1
# --dev: mount this checkout's example/ over the image's (code newer than the image; pure
#        Python changes only, the compiled core stays the image's).
# Environment: IMAGE (jhputschke/xscape-prod:cu124-oo_rhic_run), GPUS ("0 1"), JOBS_PER_GPU (8),
#              OMP_THREADS (3), MUSIC_CUDA_SYNC (block), LBT_CACHE (host path of an LBT table cache
#              file, e.g. ~/prod_test/lbt_cache/lbt_tables_v1.bin: mounted read-only, passed as
#              LBT_TABLE_CACHE; unset: LBT parses its tables).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_HOST="$(cd "$HERE/../.." && pwd)"
EXAMPLE_IMG=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example
IMAGE=${IMAGE:-jhputschke/xscape-prod:cu124-oo_rhic_run}
GPUS=${GPUS:-"0 1"}
J=${JOBS_PER_GPU:-8}
OMP=${OMP_THREADS:-3}
SYNC=${MUSIC_CUDA_SYNC:-block}

dev=()
if [ "${1:-}" = "--dev" ]; then dev=(-v "$EXAMPLE_HOST:$EXAMPLE_IMG:ro"); shift; fi
[ $# -ge 2 ] || { sed -n '2,24p' "$0"; exit 2; }
OUTROOT=$1; shift
case $OUTROOT in /*) ;; *) echo "OUTROOT must be an absolute path" >&2; exit 2 ;; esac
mkdir -p "$OUTROOT"

read -r -a gpus <<< "$GPUS"
N=${#gpus[@]}
lbt=()
if [ -n "${LBT_CACHE:-}" ]; then
  [ -f "$LBT_CACHE" ] || { echo "LBT_CACHE: no file $LBT_CACHE" >&2; exit 2; }
  lbt=(-v "$(dirname "$LBT_CACHE"):/work/lbt_cache:ro" -e LBT_TABLE_CACHE="/work/lbt_cache/$(basename "$LBT_CACHE")")
fi
common=(--user "$(id -u):$(id -g)" -e OMP_WAIT_POLICY=passive -e MUSIC_CUDA_SYNC="$SYNC" ${lbt[@]+"${lbt[@]}"}
        -v "$OUTROOT:$OUTROOT" ${dev[@]+"${dev[@]}"} -w "$EXAMPLE_IMG/analysis/OO-RHIC")
args=(run "$@" --out "$OUTROOT" -j "$J" --mps --omp "$OMP")

for a in "$@"; do
  if [ "$a" = "--list" ]; then              # no GPU, one shard after the other, in the foreground
    for k in "${!gpus[@]}"; do
      echo "== shard $k/$N (GPU ${gpus[$k]})"
      docker run --rm "${common[@]}" "$IMAGE" python oo_bias.py "${args[@]}" --shard "$k/$N"
    done
    exit 0
  fi
done

for k in "${!gpus[@]}"; do
  g=${gpus[$k]}
  name=oo_bias_gpu$g
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    echo "container $name exists (running or not removed): docker rm -f $name, or wait" >&2; exit 1
  fi
  docker run -d --name "$name" --gpus "\"device=$g\"" "${common[@]}" "$IMAGE" \
      python oo_bias.py "${args[@]}" --shard "$k/$N" > /dev/null
  echo "GPU $g: shard $k/$N, docker logs -f $name"
done
echo "outputs: $OUTROOT/<sample>/ (logs run_seed<N>.log, summaries *_seed<N>.json)"
