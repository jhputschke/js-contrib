#!/usr/bin/env bash
# example/bes_smash_ic/run_bes_smash_ic.sh
#
# One job of the staged SMASH -> MUSIC -> iSS -> SMASH chain (X-SCAPE's
# SMASHInitialCondition) at beam-energy-scan energies, with B, Q, S through the
# hydro (NEOS-4D, EOS 20) on the GPU (MUSIC4GPU, CUDA build).
#
#   ./run_bes_smash_ic.sh [options] OUTDIR
#     --build DIR        X-SCAPE build tree with SMASHInitialCondition
#                        (default: $XSCAPE/build_gpu, XSCAPE from the script's location)
#     --events N         events (default 1)
#     --seed S           random seed (default 1; same seed, same collision)
#     --sqrts E          sqrt(s_NN) in GeV (default 19.6)
#     --b MIN MAX        impact parameter range in fm, sampled b dB (default 6 8)
#     --cpu              MUSIC on the CPU (MUSIC_FORCE_CPU=1), for comparisons
#     --diffusion        net baryon diffusion on (C_B = 0.4)
#     --checks           MUSIC's per-step diagnostics (not beastMode: N_B, N_Q, N_S
#                        on the grid every step; slower) and iSS's charge check
#     --samples K        iSS samples per surface (default 1; SMASH takes the first,
#                        the charge check averages over all K)
#     --pythia DIR       prefix of the Pythia 8 that SMASH was built with, when it is
#                        not the one X-SCAPE links (SMASH-3.2.2 needs exactly 8.315):
#                        preloaded, and its xmldoc used
#     --threads T        OMP_NUM_THREADS (default 16)
#     --dynamic          dynamic fluidization (X-SCAPE branch trackB): SMASH runs
#                        the whole event next to a Cartesian MUSIC from t = 0;
#                        hadrons that pass the arXiv:2507.19389 criterion become
#                        hydro sources, particlized hadrons enter SMASH every
#                        step; grid x, y, z +-12 fm in 0.4 fm cells; prints the
#                        totals of the final state (conservation.py)
#     --no-drain         with --dynamic: keep the particlized fluid on the grid
#                        (<BDM><drain_particlized_fluid>0, as before 2026-11-17;
#                        particlizes it again when the dense region grows back)
#
# Output in OUTDIR/<tag>/: the final-state hadrons (<tag>_final_state_hadrons.dat),
# run.log (stdout and stderr, each line with the seconds since the start),
# time.log (wall time, peak memory, exit status; timed_run.py), the XML used, and with --checks charges.txt from
# bqs_balance.py (B, Q, S from SMASH into the hydro, on the grid, through the
# surface, in the hadrons). <tag> = s<sqrts>_seed<seed>[_cpu][_diff][_dyn].
#
# Needs: a CUDA build of X-SCAPE with -DUSE_SMASH=ON (SMASH-3.2.2), and the
# NEOS-4D tables in music4gpu/EOS/neos4D (cd external_packages/music4gpu/EOS &&
# bash download_Neos4D.sh) -- see README.md.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XSCAPE="$(cd "$here/../../../../../.." && pwd)"
build="$XSCAPE/build_gpu"
events=1 seed=1 sqrts=19.6 bmin=6 bmax=8 cpu=0 diff=0 checks=0 samples=1
pythia="" threads=16 dynamic=0 drain=1
usage() { sed -n '7,39p' "$0"; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --build) build="$(cd "$2" && pwd)"; shift 2 ;;
    --events) events="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --sqrts) sqrts="$2"; shift 2 ;;
    --b) bmin="$2"; bmax="$3"; shift 3 ;;
    --cpu) cpu=1; shift ;;
    --diffusion) diff=1; shift ;;
    --checks) checks=1; shift ;;
    --samples) samples="$2"; shift 2 ;;
    --pythia) pythia="$(cd "$2" && pwd)"; shift 2 ;;
    --threads) threads="$2"; shift 2 ;;
    --dynamic) dynamic=1; shift ;;
    --no-drain) drain=0; shift ;;
    -h|--help) usage ;;
    -*) echo "unknown option $1" >&2; usage ;;
    *) break ;;
  esac
done
[ $# -eq 1 ] || usage
outdir="$1"
exe="$build/SMASHInitialCondition"
[ -x "$exe" ] || { echo "no $exe: build X-SCAPE with -DUSE_SMASH=ON" >&2; exit 1; }
neos="$XSCAPE/external_packages/music4gpu/EOS/neos4D/neos4d_p_b.dat"
[ -e "$neos" ] || { echo "no NEOS-4D tables ($neos): see README.md" >&2; exit 1; }

tag="s${sqrts}_seed${seed}"
[ $cpu = 1 ] && tag="${tag}_cpu"
[ $diff = 1 ] && tag="${tag}_diff"
[ $dynamic = 1 ] && tag="${tag}_dyn"
[ $dynamic = 1 ] && [ $drain = 0 ] && tag="${tag}_nodrain"
work="$outdir/$tag"
mkdir -p "$work"
work="$(cd "$work" && pwd)"

# the run directory: MUSIC and iSS look for EOS/, iSS_tables/ and music_input in
# it, and the MUSIC wrapper rewrites music_input (so each job has its own)
ln -sfn "$build/EOS" "$work/EOS"
ln -sfn "$build/iSS_tables" "$work/iSS_tables"
cp "$build/music_input" "$work/music_input"
beast=$([ $checks = 1 ] && echo 0 || echo 1)
sed -e "s#@XSCAPE@#$XSCAPE#g" -e "s#@OUT@#$tag#" -e "s#@SEED@#$seed#" \
    -e "s#@NEVENTS@#$events#" -e "s#@SQRTS@#$sqrts#" -e "s#@BMIN@#$bmin#" \
    -e "s#@BMAX@#$bmax#" -e "s#@BEAST@#$beast#" -e "s#@DIFF@#$diff#" \
    -e "s#@NSAMPLES@#$samples#" -e "s#@CHARGECHECK@#$checks#" \
    "$here/jetscape_user_bes_smash_ic.xml" > "$work/$tag.xml"
if [ $dynamic = 1 ]; then
  # the staged template -> dynamic fluidization: SMASH (the initial-state
  # module) runs to 40 fm/c, MUSIC in (t, x, y, z) from t = 0 on a coarser,
  # longer grid, and the BDM exchanges hadrons every clock tick
  sed -i -e 's#</BDM>#  <dynamic_fluidization>1</dynamic_fluidization>\n  </BDM>#' \
      -e 's#<grid_max_x> 15 </grid_max_x>#<grid_max_x> 12 </grid_max_x>#' \
      -e 's#<grid_max_y> 15 </grid_max_y>#<grid_max_y> 12 </grid_max_y>#' \
      -e 's#<grid_max_z> 7 </grid_max_z>#<grid_max_z> 12 </grid_max_z>#' \
      -e 's#<grid_step_\([xyz]\)> 0\.[12] </grid_step_\1>#<grid_step_\1> 0.4 </grid_step_\1>#' \
      -e 's#<end_time>1000</end_time>#<end_time>40</end_time>#' \
      -e 's#<Delta_Time>0.002</Delta_Time>#<Delta_Time>0.05</Delta_Time>#' \
      -e 's#<CartesianHydro> false </CartesianHydro>#<CartesianHydro> true </CartesianHydro>#' \
      -e 's#<Initial_time_tau_0>0.5</Initial_time_tau_0>#<Initial_time_tau_0>0.0</Initial_time_tau_0>#' \
      "$work/$tag.xml"
  n=$(grep -cE "<dynamic_fluidization>1<|grid_max_[xyz]> 12 <|grid_step_[xyz]> 0.4 <|<end_time>40<|<Delta_Time>0.05<|CartesianHydro> true|tau_0>0.0<" "$work/$tag.xml")
  [ "$n" = 11 ] || { echo "--dynamic: the template changed ($n of 11 edits applied)" >&2; exit 1; }
  if [ $drain = 0 ]; then
    sed -i 's#</BDM>#  <drain_particlized_fluid>0</drain_particlized_fluid>\n  </BDM>#' "$work/$tag.xml"
  fi
fi

export OMP_NUM_THREADS="$threads"
# SMASH's tabulated integrals, computed once and shared by the jobs of OUTDIR
# (about 12 s per job otherwise; needs X-SCAPE's SmashTabulations.h)
export SMASH_TABULATIONS="${SMASH_TABULATIONS:-$(cd "$outdir" && pwd)/smash_tabulations}"
export HYDROPROGRAMPATH="$work"
[ $cpu = 1 ] && export MUSIC_FORCE_CPU=1
if [ -n "$pythia" ]; then
  export LD_PRELOAD="$pythia/lib/libpythia8.so${LD_PRELOAD:+:$LD_PRELOAD}"
  export PYTHIA8DATA="$pythia/share/Pythia8/xmldoc"
fi

cd "$work"
echo "[bes_smash_ic] $tag in $work"
python3 "$here/timed_run.py" run.log time.log -- \
  "$exe" "$tag.xml" "$XSCAPE/config/jetscape_main.xml" || true
grep -E "Elapsed|Maximum resident|Exit status" time.log
if [ $checks = 1 ]; then
  python3 "$here/bqs_balance.py" run.log "${tag}_final_state_hadrons.dat" \
    | tee charges.txt
fi
if [ $dynamic = 1 ]; then
  grep -E "Dynamic fluidization, (event totals|exchange)|SMASH IC, dynamic|MUSIC, (time-stepped|fluid:)|iSS, (time-stepped|surface fluxes)" run.log \
    | sed -E 's/.*(Dynamic|SMASH IC|MUSIC,|iSS,)/\1/; s/\x1b\[[0-9;]*m//g'
  python3 "$here/conservation.py" "${tag}_final_state_hadrons.dat" --sqrts "$sqrts" \
    | tee conservation.txt
fi
