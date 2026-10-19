#!/bin/bash
# The data of the Z+jet test (README.md): 96 Pb+Pb 5.02 TeV 0-30 % Z+jet events at hadron
# level, the pp (vacuum) reference, CMS 2507.09307's HEPData record and one event's movies,
# into ../../runs/zjet_setup/data/; and (2026-10-16) the samples of the campaign set-up checks: the vacuum references
# without the parton cut and with the campaign's windows, with all particles for the Z+jet
# analysis, CMS 1702.01060's HEPData, and Pb+Pb jobs with the campaign's arguments.
#
#   ./make_data.sh [pbpb|vacuum|hepdata|movie|all]   the test (default all)
#   ./make_data.sh [vacuum_nocut|vacuum_zh|vacuum_all|hepdata_zjet|pbpb_campaign|checks]
#                                                    the checks (checks = all five)
#   OUT=<dir> writes elsewhere than runs/zjet_setup/data/.
#
# These are the commands of 2026-10-15 on the GB10 (conda env js_fno, X-SCAPE built in
# build_gpu with the PyJetscape bindings), with the paths made relative.  Pb+Pb: two jobs
# in parallel, 15-18 min and 14-16 GB each; vacuum: ~4 min on the CPU; movie: 3 min and
# 17 GB for the event, 1.5 min for the two renders.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
PROD=$HERE/../../../prod_AuAu_0_10_jet         # run_prod_jet.py, run_prod_vacuum.py, hadronize.py
CONFIG=$HERE/../../../config                   # PbPb_0_30_ZJet.xml, grid_PbPb.yaml
VIS=$HERE/../../../../../Visualization         # wake_pyvista.py, hydro_jet_particles_pyvista.py
OUT=${OUT:-$HERE/../../runs/zjet_setup/data}
WHAT=${1:-all}
WINDOWS=20-30,30-40,40-60,60-80,80-120,120-200        # the test's (the campaigns' until 2026-10-16)
ZH_WINDOWS=20-30,30-35,35-45,45-55,55-70,70-200      # the campaigns' since 2026-10-16
mkdir -p "$OUT/logs"

if [[ $WHAT == pbpb || $WHAT == all ]]; then
  # 2 jobs x 48 events: per job 2 hydro backgrounds x 6 pTHat windows x 4 jets.  Hadrons
  # only (no pair / particlize files): 40 iSS samples of the jet and the background legs,
  # sampled correlated, 10 fragmentations; charged hadrons with |eta| < 4.
  cd "$PROD"
  export MUSIC_CUDA_SYNC=block OMP_NUM_THREADS=6
  for s in 101 102; do
    python run_prod_jet.py --events 48 --seed $s --user-xml "$CONFIG/PbPb_0_30_ZJet.xml" \
      --grid "$CONFIG/grid_PbPb.yaml" --lepton-ptmin 20 --lepton-etamax 2.5 --seed-registry none \
      --pthat-bins $WINDOWS --jets-per-bin 4 --parton-ymax 1.9 \
      --hadrons-only --oversample 40 --n-frag 10 --correlated \
      --charged --eta-max 4 --keep-primaries 0 --outdir "$OUT/pbpb" > "$OUT/logs/pbpb_seed$s.log" 2>&1 &
  done
  wait
fi

if [[ $WHAT == vacuum || $WHAT == all ]]; then
  # pp: the same XML's PythiaGun and Matter in vacuum, 1,000 events per window
  cd "$PROD"
  OMP_NUM_THREADS=2 python run_prod_vacuum.py --events 6000 --seed 901 \
    --user-xml "$CONFIG/PbPb_0_30_ZJet.xml" --pthat-bins $WINDOWS --parton-ymax 1.9 \
    --lepton-ptmin 20 --lepton-etamax 2.5 --n-frag 10 --charged --eta-max 4 \
    --outdir "$OUT/vacuum" --seed-registry none > "$OUT/logs/vacuum_seed901.log" 2>&1
fi

if [[ $WHAT == hepdata || $WHAT == all ]]; then
  # CMS 2507.09307 (CMS-HIN-23-006), HEPData ins2945894 v1 (doi:10.17182/hepdata.156186)
  mkdir -p "$OUT/hepdata" && cd "$OUT/hepdata"
  curl -sL "https://www.hepdata.net/record/156186?format=json&light=true" -o record.json
  curl -sL "https://www.hepdata.net/download/submission/ins2945894/1/yaml" \
    -o HEPData-ins2945894-v1-yaml.tar.gz
  tar xzf HEPData-ins2945894-v1-yaml.tar.gz       # -> HEPData-ins2945894-v1-yaml/
fi

if [[ $WHAT == movie || $WHAT == all ]]; then
  # 2 events on one background with the full output: the pair file <stem>.h5 (hydro
  # evolution + shower/) and the particlize file; pTHat 80-120 GeV, Z and hardest parton
  # with |y| < 1
  cd "$PROD"
  export MUSIC_CUDA_SYNC=block OMP_NUM_THREADS=6
  python run_prod_jet.py --events 2 --seed 201 --user-xml "$CONFIG/PbPb_0_30_ZJet.xml" \
    --grid "$CONFIG/grid_PbPb.yaml" --lepton-ptmin 20 --lepton-etamax 2.5 --seed-registry none \
    --pthat-min 80 --pthat-max 120 --parton-ymax 1.0 --trigger-ymax 1.0 \
    --write-particlize both --outdir "$OUT/movie" > "$OUT/logs/movie_seed201.log" 2>&1
  # their hadrons, next to the particlize file
  python hadronize.py "$OUT/movie/AuAu_0_10_jet_seed0201_particlize.h5" --oversample 4 \
    --n-frag 2 --correlated > "$OUT/logs/movie_hadronize.log" 2>&1
  # the movies of event 1 (Z -> mu mu, p_T^Z 109.6 GeV, y_Z -0.49) and one frame of each
  cd "$VIS"
  python wake_pyvista.py --file "$OUT/movie/AuAu_0_10_jet_seed0201.h5" --event 1 \
    --outdir "$OUT/movie" --movie zjet_wake_event1.mp4 > "$OUT/logs/movie_wake_render.log" 2>&1
  python hydro_jet_particles_pyvista.py --file "$OUT/movie/AuAu_0_10_jet_seed0201.h5" --event 1 \
    --outdir "$OUT/movie" --movie zjet_particles_event1.mp4 --rng-seed 1 \
    > "$OUT/logs/movie_particles_render.log" 2>&1
  python - "$OUT/movie" <<'EOF'
import sys
import imageio.v3 as iio
d = sys.argv[1]
for name, frac in (("zjet_wake_event1", 0.55), ("zjet_particles_event1", 0.8)):
    frames = iio.imread(f"{d}/{name}.mp4", index=None)
    iio.imwrite(f"{d}/{name}_frame.png", frames[int(frac * (len(frames) - 1))])
EOF
fi
# ── the campaign set-up checks (2026-10-16) ───────────────────────────────────────────
vac() {   # name seed events windows [extra args]: a vacuum reference without the parton cut
  local name=$1 seed=$2 n=$3 win=$4; shift 4
  cd "$PROD"
  OMP_NUM_THREADS=2 python run_prod_vacuum.py --events "$n" --seed "$seed" \
    --user-xml "$CONFIG/PbPb_0_30_ZJet.xml" --pthat-bins "$win" --lepton-ptmin 20 \
    --lepton-etamax 2.5 --n-frag 10 "$@" --outdir "$OUT/$name" --seed-registry none \
    > "$OUT/logs/${name}_seed$seed.log" 2>&1
}
if [[ $WHAT == vacuum_nocut || $WHAT == checks ]]; then
  # the test's windows without the parton cut (parton_cut_check.py, window_layout.py)
  vac vacuum_nocut 901 6000 "$WINDOWS" --charged --eta-max 4
fi
if [[ $WHAT == vacuum_zh || $WHAT == checks ]]; then
  # the campaign's windows without the parton cut, charged (parton_cut_check.py)
  vac vacuum_zh 901 6000 "$ZH_WINDOWS" --charged --eta-max 4
fi
if [[ $WHAT == vacuum_all || $WHAT == checks ]]; then
  # the campaign's windows and storage, all particles: the pp reference of the Z+jet
  # analysis (zjet_pp_jets.py), 2,000 events per window, ~8 min
  vac vacuum_all 902 12000 "$ZH_WINDOWS" --eta-max 2.6 --no-x
fi
if [[ $WHAT == hepdata_zjet || $WHAT == checks ]]; then
  # CMS 1702.01060 (CMS-HIN-15-013, Z+jet), HEPData ins1512112 v1
  mkdir -p "$OUT/hepdata_zjet" && cd "$OUT/hepdata_zjet"
  curl -sL "https://www.hepdata.net/download/submission/ins1512112/1/yaml" \
    -o HEPData-ins1512112-v1-yaml.tar.gz
  tar xzf HEPData-ins1512112-v1-yaml.tar.gz       # -> HEPData-ins1512112-v1-yaml/
fi
if [[ $WHAT == pbpb_campaign || $WHAT == checks ]]; then
  # one 24-event Pb+Pb job (one background) with the campaign's arguments, all particles and
  # charged only: disk per event, time and memory, and the seed-101 reference values of
  # utils/zjet/zjet_3090_test.sh report (~6 min each, run one after the other)
  cd "$PROD"
  for v in pbpb_campaign: pbpb_campaign_charged:--charged; do
    name=${v%%:*}; extra=${v#*:}
    MUSIC_CUDA_SYNC=block OMP_NUM_THREADS=6 /usr/bin/time -v python run_prod_jet.py --events 24 \
      --seed 101 --user-xml "$CONFIG/PbPb_0_30_ZJet.xml" --grid "$CONFIG/grid_PbPb.yaml" \
      --seed-registry none --pthat-bins "$ZH_WINDOWS" --jets-per-bin 4 --lepton-ptmin 20 \
      --lepton-etamax 2.5 --hadrons-only --oversample 40 --n-frag 40 --correlated $extra \
      --eta-max 2.6 --no-x --hadron-keep-bits-p 12 --hadron-keep-bits-x 8 --keep-primaries 0 \
      --outdir "$OUT/$name" > "$OUT/logs/$name.log" 2>&1
  done
fi
echo "data in $OUT"
