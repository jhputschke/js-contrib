#!/usr/bin/env bash
# zjet_3090_campaign.sh -- one Z+jet campaign (10,000 Pb+Pb 5.02 TeV events) on the 2 x RTX 3090
# box, through ~/prod_test/launch_2gpu.sh.  Same campaign layout and seeds as
# zjet_slurm_campaign.sh, so campaigns from both machines combine (docs/Plans/PLAN_ZJET_CMS10x.md).
#
#   ./zjet_3090_campaign.sh 0            # campaign zjet_c00: 100 jobs x 100 events, ~9 h
#   ./zjet_3090_campaign.sh 0 10         # smoke test: 10 jobs x 100 events (~1,000 events)
#   ./zjet_3090_campaign.sh 0            # rerun the same command to resume
#
# 6 jobs per GPU (Pb+Pb jobs ~8-9 GB; 20 x 8.5 GB would not fit 125 GB), OMP 4.  ~1,000-1,300
# events/h expected (plan section 6); replace with the benchmark's numbers.
#
# The inputs are the image's own docs/PbPb_5020 files (Z+jet XML, Pb+Pb hadronization with
# eCMforHadronization 5020, the Pb+Pb output grid), so they always match the image's code;
# XML_DIR=/work/xml takes copies from ~/prod_test/xml instead.
set -u
C=${1:?campaign number 0..9}
NJOBS=${2:-100}
case $C in [0-9]) ;; *) echo "campaign must be 0..9, got '$C'" >&2; exit 2 ;; esac
EVENTS=100                                     # 5 backgrounds x (4 windows x 5 jets)
SEED0=$(( 1 + 1000 * C ))                      # job i of campaign C has seed SEED0 + i
CAMP=$(printf "zjet_c%02d" "$C")
OUT=${OUT:-/data/$CAMP}
XML_DIR=${XML_DIR:-/opt/X-SCAPE/external_packages/js-contrib/docs/PbPb_5020}
export IMAGE=${IMAGE:?set IMAGE to the Z+jet image, pinned by digest (jhputschke/xscape-prod@sha256:...)}
export JOBS_PER_GPU=${JOBS_PER_GPU:-6}
export OMP_THREADS=${OMP_THREADS:-4}
export MUSIC_CUDA_SYNC=block

exec "$HOME/prod_test/launch_2gpu.sh" \
  --lbt-cache /work/lbt_cache/lbt_tables_v1.bin --stagger 12 --min-free 100G \
  "$NJOBS" "$EVENTS" "$SEED0" "$OUT" --campaign "$CAMP" \
  --user-xml "$XML_DIR/PbPb_0_10_ZJet.xml" --hadronize-xml "$XML_DIR/hadronize_5020.xml" \
  --grid "$XML_DIR/grid_PbPb.yaml" \
  --pthat-bins 40-60,60-80,80-120,120-200 --jets-per-bin 5 --parton-ymax 1.9 \
  --trigger-ymax 2.4 --lepton-ptmin 20 --lepton-etamax 2.4 \
  --hadrons-only --oversample 40 --n-frag 40 --oversample-bg per-pthat-bin --correlated \
  --charged --eta-max 2.6 --no-x --hadron-keep-bits-p 12 --hadron-keep-bits-x 8 --keep-primaries 0 \
  > "$OUT.log" 2>&1
