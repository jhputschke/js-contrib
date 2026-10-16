#!/usr/bin/env bash
# zjet_3090_test.sh -- the Z+jet image on the 2 x RTX 3090 box before the PRs and campaign 0
# (PLAN_ZJET_CMS10x.md 5a step 10, utils/zjet/README.md "Before campaign 0").  Run it in
# ~/prod_test (next to launch_2gpu.sh), one step at a time:
#
#   export IMAGE=jhputschke/xscape-prod:cu124-zjet
#   ./zjet_3090_test.sh info      # pull, digest, commits, --help, dry run           (~2 min)
#   ./zjet_3090_test.sh bench 3   # 3 jobs per GPU x 24 Pb+Pb Z+jet events (one       (~25 min)
#                                 # background each), the campaign arguments: per-job peak
#                                 # memory, host peak, makespan, events/h
#   ./zjet_3090_test.sh bench 4   # the same at 4 per GPU, only if bench 3's host peak left
#                                 # room for 2 more jobs (report says so)
#   ./zjet_3090_test.sh report    # the numbers of every bench run, and seed 101's first events
#                                 # against the GB10's (the same Pythia and Glauber draws?)
#   ./zjet_3090_test.sh smoke     # zjet_3090_campaign.sh 0 10: ~1,000 events (~2 h)
#
# OUT (default /data/zjet_test) holds everything; LBT_CACHE (default
# /work/lbt_cache/lbt_tables_v1.bin) is the campaigns' table cache.  The image must be pinned:
# the digest info prints goes into the campaign log and IMAGE=...@sha256:... for campaign 0.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP=${1:?step: info | bench J | report | smoke}
IMAGE=${IMAGE:?set IMAGE, e.g. jhputschke/xscape-prod:cu124-zjet}
OUT=${OUT:-/data/zjet_test}
LBT_CACHE=${LBT_CACHE:-/work/lbt_cache/lbt_tables_v1.bin}
D=/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/config
LAUNCH=${LAUNCH:-$HOME/prod_test/launch_2gpu.sh}
ARGS=(--user-xml "$D/PbPb_0_30_ZJet.xml" --grid "$D/grid_PbPb.yaml"
      --pthat-bins 20-30,30-40,40-60,60-80,80-120,120-200 --jets-per-bin 4 --parton-ymax 1.9
      --lepton-ptmin 20 --lepton-etamax 2.5
      --hadrons-only --oversample 40 --n-frag 40 --correlated
      --charged --eta-max 2.6 --no-x --hadron-keep-bits-p 12 --hadron-keep-bits-x 8
      --keep-primaries 0)
mkdir -p "$OUT" || exit 1
run_img() { docker run --rm --user "$(id -u):$(id -g)" -v "$OUT:$OUT" "$IMAGE" "$@"; }

case $STEP in
info)
  docker pull "$IMAGE" || exit 1
  digest=$(docker inspect --format '{{index .RepoDigests 0}}' "$IMAGE")
  echo "image: $IMAGE = $digest" | tee "$OUT/image.txt"
  run_img cat /opt/X-SCAPE/BUILD_INFO.txt | tee -a "$OUT/image.txt"
  echo "--- run_prod_jet.py --help (Z options):"
  run_img python run_prod_jet.py --help | grep -E -- "--trigger|--lepton-|--hadrons-only" | head -6
  echo "--- dry run of a campaign job (no GPU):"
  run_img python run_prod_jet.py --events 96 --seed 1 "${ARGS[@]}" --seed-registry none \
    --dry-run --outdir "$OUT/dry" > "$OUT/dry.log" 2>&1
  ok=1
  for want in "trigger Z (lepton pT > 20 GeV, lepton |eta| < 2.5)" "EOS 91" \
              "eCMforHadronization 5020" "reuse 24" "HADRONS ONLY"; do
    if grep -qF "$want" "$OUT/dry.log"; then echo "  ok: $want"; else echo "  MISSING: $want"; ok=0; fi
  done
  if grep -q "<b_max> *8.57" "$OUT"/dry/*.xml; then echo "  ok: <b_max> 8.57 in the job XML"
  else echo "  MISSING: <b_max> 8.57 in $OUT/dry/*.xml"; ok=0; fi
  [ $ok = 1 ] && echo "info: all checks passed" || { echo "info: see $OUT/dry.log"; exit 1; }
  ;;
bench)
  J=${2:?bench J (jobs per GPU)}
  B=$OUT/bench_j$J
  mkdir -p "$B" || exit 1
  [ -x "$LAUNCH" ] || { echo "no $LAUNCH (set LAUNCH=...)" >&2; exit 1; }
  # host memory used, every 5 s, while the jobs run
  ( while :; do echo "$(date +%s) $(free -m | awk '/^Mem:/ {print $3}')"; sleep 5; done ) \
    > "$B/hostmem.log" &
  mon=$!
  t0=$(date +%s)
  JOBS_PER_GPU=$J OMP_THREADS=${OMP_THREADS:-6} MUSIC_CUDA_SYNC=block IMAGE=$IMAGE \
    "$LAUNCH" --lbt-cache "$LBT_CACHE" --stagger 12 --min-free 100G \
    $((2 * J)) 24 101 "$B" "${ARGS[@]}" > "$B/launch.log" 2>&1
  rc=$?
  t1=$(date +%s)
  kill $mon 2>/dev/null
  echo "$t0 $t1 $rc" > "$B/makespan.txt"
  echo "bench J=$J: exit $rc, makespan $((t1 - t0)) s; ./zjet_3090_test.sh report"
  ;;
report)
  docker run --rm -i --user "$(id -u):$(id -g)" -v "$OUT:$OUT" "$IMAGE" python - "$OUT" <<'EOF'
import glob, json, os, re, sys
import numpy as np
out = sys.argv[1]
for b in sorted(glob.glob(os.path.join(out, "bench_j*"))):
    J = int(b.rsplit("j", 1)[1])
    try:
        t0, t1, rc = map(int, open(os.path.join(b, "makespan.txt")).read().split())
    except OSError:
        print(f"{b}: not finished"); continue
    jobs = []
    for js in sorted(glob.glob(os.path.join(b, "gpu*", "*.json"))):
        d = json.load(open(js))
        log = js[:-5] + ".log"
        txt = re.sub(r"\x1b\[[0-9;]*m", "", open(log, errors="replace").read()) \
            if os.path.exists(log) else ""
        mb = [int(x) for x in re.findall(r"\[Info\]\s+(\d+)MB", txt)]
        jobs.append((d.get("events_written", 0), d.get("wall_s", np.nan), max(mb) / 1e3 if mb else np.nan,
                     d.get("pthat_bin_acceptance"), d.get("events_hit_edge"), d.get("droplets_total")))
    host = np.loadtxt(os.path.join(b, "hostmem.log"), ndmin=2)
    base, peak = host[0, 1] / 1e3, host[:, 1].max() / 1e3
    ev = sum(j[0] for j in jobs)
    ms = t1 - t0
    pk = [j[2] for j in jobs]
    print(f"== {J} jobs per GPU ({len(jobs)} jobs, exit {rc}): {ev} events in {ms} s -> "
          f"{ev / ms * 3600:.0f} events/h by makespan")
    print(f"   per job: {np.nanmean([j[1] / max(j[0], 1) for j in jobs]):.1f} s per event (wall_s), "
          f"peak memory {np.nanmin(pk):.1f}-{np.nanmax(pk):.1f} GB (ru_maxrss)")
    print(f"   host: {base:.0f} GB before, peak {peak:.0f} GB of {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9:.0f} "
          f"-> room for ~{int((118 - peak) // max(np.nanmax(pk), 1))} more jobs of this size (118 GB usable)")
    print(f"   droplets per job {[j[5] for j in jobs]}, edge hits {[j[4] for j in jobs]}, "
          f"acceptance (job 1) {jobs[0][3] if jobs else None}")
# seed 101's first events against the GB10 (zjet_test, 2026-10-15): Pythia and 3dMCGlauber
# draws are CPU-side, so they should be the same on both machines
ref = [(0, 29.618, +0.3052), (1, 36.936, -1.0004), (2, 62.470, -1.6314), (3, 95.602, +1.3301),
       (4, 111.481, -0.4506), (5, 128.379, +0.5013)]
f = [js[:-5] + "_hadrons_jet_frag.h5" for js in sorted(glob.glob(os.path.join(out, "bench_j*", "gpu*", "*.json")))
     if json.load(open(js)).get("seed") == 101]
f = [x for x in f if os.path.exists(x)]
if f:
    import h5py
    with h5py.File(f[0], "r") as h:
        u = h["units"]
        zpt, zy = u["z_pt"][:6], u["z_y"][:6]
        nc = u["n_coll"][0] if "n_coll" in u else None
    print(f"== seed 101 against the GB10 (N_coll there 443, here {nc}):")
    for (k, p, y), a, b_ in zip(ref, zpt, zy):
        same = abs(a - p) < 1e-3 and abs(b_ - y) < 1e-4
        print(f"   event {k}: Z pT {a:8.3f} (GB10 {p:8.3f}), y {b_:+.4f} (GB10 {y:+.4f})  "
              f"{'same' if same else 'DIFFERENT'}")
EOF
  ;;
smoke)
  C=${C:-0}
  [ -x "$HERE/zjet_3090_campaign.sh" ] || { echo "no $HERE/zjet_3090_campaign.sh" >&2; exit 1; }
  OUT=${SMOKE_OUT:-/data/zjet_smoke} IMAGE=$IMAGE exec "$HERE/zjet_3090_campaign.sh" "$C" 10
  ;;
*) echo "unknown step '$STEP' (info | bench J | report | smoke)" >&2; exit 2 ;;
esac
