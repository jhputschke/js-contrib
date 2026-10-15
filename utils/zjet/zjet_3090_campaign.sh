#!/usr/bin/env bash
# zjet_3090_campaign.sh -- one Z+jet campaign (9,600 Pb+Pb 5.02 TeV 0-30 % events) on the 2 x RTX 3090
# box, through ~/prod_test/launch_2gpu.sh.  Same campaign layout and seeds as
# zjet_slurm_campaign.sh, so campaigns from both machines combine (docs/Plans/PLAN_ZJET_CMS10x.md).
#
#   ./zjet_3090_campaign.sh 0            # campaign zjet_c00: 100 jobs x 96 events, ~9 h
#   ./zjet_3090_campaign.sh 0 10         # smoke test: 10 jobs x 96 events (~1,000 events)
#   ./zjet_3090_campaign.sh 0            # rerun the same command to resume
#
# 6 jobs per GPU (Pb+Pb jobs ~8-9 GB; 20 x 8.5 GB would not fit 125 GB), OMP 4.  ~1,000-1,300
# events/h expected (plan section 6); replace with the benchmark's numbers.
#
# The inputs are the image's own example/config files (the Z+jet XML and the Pb+Pb output
# grid), so they always match the image's code; XML_DIR=/work/xml takes copies from
# ~/prod_test/xml instead.  The hadronization settings are the image's config/hadronize.xml,
# with eCMforHadronization the collision's (5020, from the XML's PythiaGun).
set -u
C=${1:?campaign number 0..9}
NJOBS=${2:-100}
case $C in [0-9]) ;; *) echo "campaign must be 0..9, got '$C'" >&2; exit 2 ;; esac
EVENTS=96                                      # 4 backgrounds x (6 windows x 4 jets)
SEED0=$(( 1 + 1000 * C ))                      # job i of campaign C has seed SEED0 + i
CAMP=$(printf "zjet_c%02d" "$C")
OUT=${OUT:-/data/$CAMP}
XML_DIR=${XML_DIR:-/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/config}
export IMAGE=${IMAGE:?set IMAGE to the Z+jet image, pinned by digest (jhputschke/xscape-prod@sha256:...)}
export JOBS_PER_GPU=${JOBS_PER_GPU:-6}
export OMP_THREADS=${OMP_THREADS:-4}
export MUSIC_CUDA_SYNC=block

exec "$HOME/prod_test/launch_2gpu.sh" \
  --lbt-cache /work/lbt_cache/lbt_tables_v1.bin --stagger 12 --min-free 100G \
  "$NJOBS" "$EVENTS" "$SEED0" "$OUT" --campaign "$CAMP" \
  --user-xml "$XML_DIR/PbPb_0_30_ZJet.xml" --grid "$XML_DIR/grid_PbPb.yaml" \
  --pthat-bins 20-30,30-40,40-60,60-80,80-120,120-200 --jets-per-bin 4 --parton-ymax 1.9 \
  --lepton-ptmin 20 --lepton-etamax 2.5 \
  --hadrons-only --oversample 40 --n-frag 40 --correlated \
  --charged --eta-max 2.6 --no-x --hadron-keep-bits-p 12 --hadron-keep-bits-x 8 --keep-primaries 0 \
  > "$OUT.log" 2>&1
