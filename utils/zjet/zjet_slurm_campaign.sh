#!/bin/bash
# zjet_slurm_campaign.sh -- one Z+jet campaign (9,600 Pb+Pb 5.02 TeV 0-30 % events) on NCSA Delta
# A40s, through js-contrib's utils/slurm_prod_array.sh.  Ten campaigns C=0..9 give the 96k events of
# docs/Plans/PLAN_ZJET_CMS10x.md (section 5b); one campaign has 2.7x CMS 2507.09307's Z count and
# 10x CMS Z+jet's p_T^Z > 60 count (effective, after the window and N_coll weights; plan section 2).
#
#   C=0 sbatch zjet_slurm_campaign.sh            # campaign zjet_c00: 10 A40s x 10 jobs x 96 events
#   C=1 sbatch zjet_slurm_campaign.sh            # ... up to C=9
#   C=0 NJOBS=1 sbatch zjet_slurm_campaign.sh    # smoke test: ~1,000 events = one CMS Z+jet (p_T^Z > 60)
#   C=0 sbatch zjet_slurm_campaign.sh            # resubmit the same command to resume
#
# Per task (one A40): one GPU's share of a gpuA40x4 node, 16 cores, 58 GiB; P=3 jobs at once
# (14-16 GB each: a Pb+Pb Z+jet job's peak on the GB10, docs/BENCHMARK_PbPb_vs_AuAu.md),
# OMP_NUM_THREADS = 16/3 = 5.  ~250-400 events/h per A40 (estimate, plan section 6): 960
# events per task in ~3-4.5 h (10 jobs at P=3 = 4 rounds).  TO BE CONFIRMED (benchmark, plan
# 5a step 10): set P from the measured MaxRSS.
#
# Before campaign 0: the Z+jet image pulled into $SIF (pinned by digest), the LBT cache made
# once by a 1-event job, and the 15-event benchmark (plan 5a step 10).  The inputs are the
# image's own example/config files (hadronization: its config/hadronize.xml, with the
# collision's eCMforHadronization); XML_DIR=/xml (with BINDS) takes your copies instead.
#
#SBATCH --job-name=zjet
#SBATCH --partition=gpuA40x4
#SBATCH --account=CHANGE_ME-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --output=zjet_%A_%a.out
set -u
C=${C:?set C=0..9 (the campaign number)}
case $C in [0-9]) ;; *) echo "C must be 0..9, got '$C'" >&2; exit 2 ;; esac

export SIF=${SIF:-$HOME/xscape_prod_zjet.sif}          # the Z+jet image (plan 5a, step 9)
export WORK=${WORK:-/work/hdd/$USER/xscape_prod}       # shared by all tasks and campaigns
XML_DIR=${XML_DIR:-/opt/X-SCAPE/external_packages/js-contrib/contribs/PyJetscape/example/config}
export PROD=auau
export CAMPAIGN=$(printf "zjet_c%02d" "$C")
export P=${P:-3}                                       # jobs at once per A40 (GB10 memory; confirm)
export NJOBS=${NJOBS:-10}                              # jobs per task: 10 tasks x 10 jobs = 100 jobs
export EVENTS=${EVENTS:-96}                            # 4 backgrounds x (6 windows x 4 jets)
export SEED_MODE=ranges                                # no cross-node lock needed
export SEED_BASE=$(( 1 + 1000 * C ))                   # campaigns never share a seed
export MUSIC_CUDA_SYNC=block
export TAG_PREFIX=${TAG_PREFIX:-PbPb_0_30_seed}          # PbPb_0_30_zjet_cNN_tTTT_seedNNNN.*
export LBT_CACHE=/work/lbt_cache/lbt_tables_v1_amd64.bin
export STAGGER=12
export USE_MPS=${USE_MPS:-1}                           # 0 if the GPU is in Exclusive_Process mode
export CHECK_DEPOSITION=1

export EXTRA_ARGS=${EXTRA_ARGS:-"--user-xml $XML_DIR/PbPb_0_30_ZJet.xml \
  --grid $XML_DIR/grid_PbPb.yaml \
  --pthat-bins 20-30,30-35,35-45,45-55,55-70,70-200 --jets-per-bin 4 \
  --lepton-ptmin 20 --lepton-etamax 2.5 \
  --hadrons-only --oversample 40 --n-frag 40 --correlated \
  --eta-max 2.6 --no-x --hadron-keep-bits-p 12 --hadron-keep-bits-x 8 --keep-primaries 0"}

# slurm_prod_array.sh of the same checkout: sbatch copies this script away, so look where it
# was submitted from (utils/zjet, utils or the repo top), then ~/js-contrib
if [ -z "${SCRIPT:-}" ]; then
  for SCRIPT in "${SLURM_SUBMIT_DIR:-$PWD}/../slurm_prod_array.sh" \
                "${SLURM_SUBMIT_DIR:-$PWD}/slurm_prod_array.sh" \
                "${SLURM_SUBMIT_DIR:-$PWD}/utils/slurm_prod_array.sh" \
                "$HOME/js-contrib/utils/slurm_prod_array.sh"; do
    [ -f "$SCRIPT" ] && break
  done
fi
[ -f "$SCRIPT" ] || { echo "slurm_prod_array.sh not found at $SCRIPT (set SCRIPT=...)" >&2; exit 1; }
grep -q 'TAG_PREFIX' "$SCRIPT" ||
  { echo "$SCRIPT does not pass TAG_PREFIX to the jobs (js-contrib 2026-10-16 or later)" >&2; exit 1; }
echo "campaign $CAMPAIGN, task ${SLURM_ARRAY_TASK_ID:-0}: seeds from $(( SEED_BASE + ${SLURM_ARRAY_TASK_ID:-0} * NJOBS ))"
exec bash "$SCRIPT"
