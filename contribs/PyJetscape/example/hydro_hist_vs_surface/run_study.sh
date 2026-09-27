#!/usr/bin/env bash
# example/hydro_hist_vs_surface/run_study.sh -- the whole study, resumable (README.md).
#
#   ./run_study.sh [DATA [OUT]]
#
# DATA: the production directory: *_particlize.h5 (run_prod_jet.py --write-particlize both),
#       their pair files *.h5, and the reference hadrons *_hadrons_{bulk_jet,bulk_bg,jet_frag}.h5
#       (default: /home/putschke/FNO_Hydro_Data/AuAu_0_10_h5_pair_surface_test)
# OUT:  where the variants go, one directory each (default: DATA/hist_vs_surface)
#
# For each variant: make_surfaces.py writes its particlize files, then run_hadronize.py
# hadronizes them with the reference's settings (HADRONIZE_OPTS, which must match how the
# reference hadrons were made: see DATA/*_hadronize.log), then analyze.py summarizes
# everything into OUT/summary.npz for hydro_hist_vs_surface.ipynb.
#
# Environment:
#   PYTHON          the interpreter pyjetscape_core was built for (default: python)
#   HADRONIZE_OPTS  default "--oversample 100 --n-frag 50 --correlated --keep-bits-p 12 --keep-bits-x 8"
#   HADRONIZE_J     parallel hadronize.py processes (default 4; OMP_NUM_THREADS 5 each)
#   VARIANTS        default "ref_ideal ref_no_bulk ref_no_shear hist"
#
# Re-running resumes: complete surface and hadron files are skipped.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$(realpath "${1:-/home/putschke/FNO_Hydro_Data/AuAu_0_10_h5_pair_surface_test}")"
OUT="$(realpath -m "${2:-$DATA/hist_vs_surface}")"
PY="${PYTHON:-python}"
HADRONIZE_OPTS="${HADRONIZE_OPTS:---oversample 100 --n-frag 50 --correlated --keep-bits-p 12 --keep-bits-x 8}"
HADRONIZE_J="${HADRONIZE_J:-4}"
VARIANTS="${VARIANTS:-ref_ideal ref_no_bulk ref_no_shear hist}"

mkdir -p "$OUT"
echo "run_study.sh: DATA=$DATA OUT=$OUT"
echo "  variants: $VARIANTS"
echo "  hadronize: $HADRONIZE_OPTS (-j $HADRONIZE_J)"

for v in $VARIANTS; do
    echo "=== $(date '+%F %T') $v: surfaces"
    "$PY" "$HERE/make_surfaces.py" "$v" "$DATA" --out-dir "$OUT/$v" --skip-complete
    echo "=== $(date '+%F %T') $v: hadrons"
    # shellcheck disable=SC2086
    OMP_NUM_THREADS=$(( $(nproc) / HADRONIZE_J > 0 ? $(nproc) / HADRONIZE_J : 1 )) \
        "$PY" "$HERE/../prod_AuAu_0_10_jet/run_hadronize.py" "$OUT/$v" -j "$HADRONIZE_J" \
        $HADRONIZE_OPTS
done

echo "=== $(date '+%F %T') analysis"
# shellcheck disable=SC2086
"$PY" "$HERE/analyze.py" --ref "$DATA" --variants-dir "$OUT" --variants ref $VARIANTS \
    --out "$OUT/summary.npz"
echo "=== $(date '+%F %T') done: $OUT/summary.npz"
