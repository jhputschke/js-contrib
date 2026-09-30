#!/usr/bin/env bash
# setup_analysis_env.sh -- a Python venv for analysing prod_AuAu_0_10_jet output without
# X-SCAPE: the pair, particlize and hadron .h5 files, run_h5toROOT.py and the ROOT files it
# writes (uproot, no ROOT), the analysis/ notebooks, and the Visualization scripts on the
# stored files (wake_pyvista.py, hydro_jet_particles_pyvista.py).
#
#   ./setup_analysis_env.sh [VENV_DIR] [options]      VENV_DIR default: ~/.venvs/js_analysis
#
#   --python PY          interpreter for the venv (default: python3; needs >= 3.10)
#   --uv                 make the venv and install with uv (faster; `uv` must be on PATH)
#   --system-site-packages
#                        let the venv see the site-packages of the interpreter it is made
#                        from, e.g. PyROOT of a conda env with ROOT (run from that env, so
#                        python3 is its Python). run_h5toROOT.py then writes with ROOT.
#   --no-viz             skip pyvista/vtk (requirements_viz.txt)
#   --with-fasthydro     also install FastHydro (jetscape-fasthydro), for FastHydro's own
#                        files and readers (fast_data, fasthydro.browse); not needed for the
#                        prod_AuAu_0_10_jet files. The solver also needs `pip install torch`.
#   --with-gcs           also install the Google Cloud Storage interfaces: gcsfs (gs:// for
#                        fsspec, h5py and uproot) and google-cloud-storage
#                        (requirements_gcs.txt)
#   --with-pelican       also install pelicanfs: pelican:// and osdf:// for fsspec, h5py
#                        and uproot (requirements_pelican.txt)
#   --eos-table PATH     copy MUSIC's hotQCD table from PATH (a MUSIC/X-SCAPE EOS/hotQCD
#                        directory or the hrg_hotqcd_eos_binary.dat) instead of downloading it
#   --no-eos             neither download nor copy it (no e -> T: the visualization uses a
#                        conformal EoS, jet_wake.ipynb skips the contours, analysis/wake_*.py
#                        need --eos)
#   --kernel NAME        also register a Jupyter kernel NAME for this venv (in ~/.local)
#   --check DIR          after installing, open the production files in DIR (check_env.py)
#   -h, --help           this text
#
# Re-running on an existing venv updates it (--system-site-packages switches it on for it). Nothing outside VENV_DIR is written, except the
# kernel spec with --kernel. The compiled pyjetscape_core is NOT needed; jetscape.HAS_CORE
# is then False, and everything that reads files works.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTRIBS="$(cd "$HERE/../../contribs" && pwd)"      # utils/analysis_env -> contribs
PYJETSCAPE="$CONTRIBS/PyJetscape"

VENV="$HOME/.venvs/js_analysis"
PY=python3
USE_UV=0
SYSSITE=0
VIZ=1
FASTHYDRO=0
GCS=0
PELICAN=0
EOS=download
EOS_SRC=""
KERNEL=""
CHECK_DIR=""

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
die()   { echo "setup_analysis_env: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --python)          PY="${2:?--python needs an interpreter}"; shift 2 ;;
        --uv)              USE_UV=1; shift ;;
        --system-site-packages) SYSSITE=1; shift ;;
        --no-viz)          VIZ=0; shift ;;
        --with-fasthydro)  FASTHYDRO=1; shift ;;
        --with-gcs)        GCS=1; shift ;;
        --with-pelican)    PELICAN=1; shift ;;
        --eos-table)       EOS=copy; EOS_SRC="${2:?--eos-table needs a path}"; shift 2 ;;
        --no-eos)          EOS=none; shift ;;
        --kernel)          KERNEL="${2:?--kernel needs a name}"; shift 2 ;;
        --check)           CHECK_DIR="${2:?--check needs a directory}"; shift 2 ;;
        -h|--help)         usage; exit 0 ;;
        -*)                die "unknown option $1 (see --help)" ;;
        *)                 VENV="$1"; shift ;;
    esac
done

# ── the venv ────────────────────────────────────────────────────────────────────────────
command -v "$PY" >/dev/null || die "no interpreter '$PY' (use --python)"
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || die "$PY is Python $("$PY" -c 'import platform; print(platform.python_version())'); need >= 3.10"
if [ "$USE_UV" = 1 ]; then
    command -v uv >/dev/null || die "--uv given but uv is not on PATH"
fi

VENV="$(mkdir -p "$VENV" && cd "$VENV" && pwd)"
VENV_OPTS=()
[ "$SYSSITE" = 1 ] && VENV_OPTS=(--system-site-packages)
if [ -x "$VENV/bin/python" ]; then
    echo "==> updating the venv $VENV"
    if [ "$SYSSITE" = 1 ] && grep -q '^include-system-site-packages *= *false' "$VENV/pyvenv.cfg"; then
        echo "    switching on --system-site-packages (pyvenv.cfg)"
        sed -i.bak 's/^include-system-site-packages *= *false/include-system-site-packages = true/' \
            "$VENV/pyvenv.cfg" && rm -f "$VENV/pyvenv.cfg.bak"
    fi
elif [ "$USE_UV" = 1 ]; then
    echo "==> creating the venv $VENV (uv, $PY${VENV_OPTS[*]+, ${VENV_OPTS[*]}})"
    uv venv --python "$PY" ${VENV_OPTS[@]+"${VENV_OPTS[@]}"} "$VENV"
else
    echo "==> creating the venv $VENV ($PY${VENV_OPTS[*]+, ${VENV_OPTS[*]}})"
    "$PY" -m venv ${VENV_OPTS[@]+"${VENV_OPTS[@]}"} "$VENV" \
        || die "python -m venv failed; on Debian/Ubuntu install python3-venv, or use --uv"
fi
if [ "$SYSSITE" = 1 ]; then
    base="$(sed -n 's/^home *= *//p' "$VENV/pyvenv.cfg")"
    echo "    it sees the site-packages of the Python in $base"
fi
VPY="$VENV/bin/python"

pip_install() {
    if [ "$USE_UV" = 1 ]; then uv pip install --python "$VPY" "$@"
    else "$VPY" -m pip install "$@"; fi
}

# ── packages ────────────────────────────────────────────────────────────────────────────
[ "$USE_UV" = 1 ] || "$VPY" -m pip install --upgrade --quiet pip

REQS=(-r "$HERE/requirements.txt")
EXTRA=""
[ "$VIZ" = 1 ] && { REQS+=(-r "$HERE/requirements_viz.txt"); EXTRA+=" + visualization"; }
[ "$GCS" = 1 ] && { REQS+=(-r "$HERE/requirements_gcs.txt"); EXTRA+=" + Google Cloud Storage"; }
[ "$PELICAN" = 1 ] && { REQS+=(-r "$HERE/requirements_pelican.txt"); EXTRA+=" + Pelican/OSDF"; }
echo "==> installing the analysis packages$EXTRA"
pip_install "${REQS[@]}"

# PyJetscape's Python package (jetscape.*: the HDF5 readers), editable from this checkout.
# Its dependencies are all in requirements.txt, so the heavy pyproject list is not pulled.
echo "==> installing PyJetscape (editable, $PYJETSCAPE)"
pip_install --no-deps -e "$PYJETSCAPE"

if [ "$FASTHYDRO" = 1 ]; then
    echo "==> installing FastHydro (editable, $CONTRIBS/FastHydro)"
    pip_install -e "$CONTRIBS/FastHydro[viz]"
fi

# ── MUSIC's hotQCD EoS table ────────────────────────────────────────────────────────────
# MUSIC pair files store e, not T. jet_wake.ipynb, analysis/wake_*.py and the visualization
# convert e -> T with MUSIC's EOS 9 table, from $MUSIC_EOS_TABLE (the .dat file itself).
# Same source as MUSIC's EOS/download_hotQCD.sh.
EOS_DIR="$VENV/share/music_eos/hotQCD"
EOS_NAME="hrg_hotqcd_eos_binary.dat"
EOS_MD5="9ac35d8387c442d168a5ca93a7864dca"
EOS_URL="https://api.bitbucket.org/2.0/repositories/wayne_state_nuclear_theory/hotqcd/src/main/$EOS_NAME"
EOS_TABLE="$EOS_DIR/$EOS_NAME"

md5_of() { "$VPY" -c 'import hashlib,sys; print(hashlib.md5(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }

eos_ok=0
if [ "$EOS" != none ]; then
    mkdir -p "$EOS_DIR"
    if [ -f "$EOS_TABLE" ] && [ "$(md5_of "$EOS_TABLE")" = "$EOS_MD5" ]; then
        eos_ok=1
    else
        got=0
        if [ "$EOS" = copy ]; then
            src="$EOS_SRC"; [ -d "$src" ] && src="$src/$EOS_NAME"
            if [ -f "$src" ]; then cp -L "$src" "$EOS_TABLE.part" && got=1
            else echo "  [!] no $EOS_NAME at $EOS_SRC"; fi
        else
            echo "==> downloading MUSIC's $EOS_NAME (3.2 MB)"
            curl -fsSL -o "$EOS_TABLE.part" "$EOS_URL" && got=1 || echo "  [!] the download failed"
        fi
        if [ "$got" = 1 ] && [ "$(md5_of "$EOS_TABLE.part")" = "$EOS_MD5" ]; then
            mv "$EOS_TABLE.part" "$EOS_TABLE"; eos_ok=1
        elif [ "$got" = 1 ]; then
            echo "  [!] $EOS_NAME has the wrong checksum; not installed"
        fi
        rm -f "$EOS_TABLE.part"
    fi
fi

# export it from bin/activate; the block is rewritten on every run
MARK="# js_analysis: MUSIC_EOS_TABLE"
"$VPY" - "$VENV/bin/activate" "$MARK" <<'PY'
import sys
path, mark = sys.argv[1], sys.argv[2]
lines = open(path).read().split("\n")
out, i = [], 0
while i < len(lines):
    if lines[i] == mark:
        i += 2                                  # the mark and its export line
        continue
    out.append(lines[i])
    i += 1
while out and out[-1] == "":
    out.pop()
open(path, "w").write("\n".join(out) + "\n")
PY
if [ "$eos_ok" = 1 ]; then
    printf '\n%s\nexport MUSIC_EOS_TABLE="%s"\n' "$MARK" "$EOS_TABLE" >> "$VENV/bin/activate"
else
    echo "  [!] no MUSIC EoS table: no e -> T for the pair files (see --eos-table)"
fi

# ── Jupyter kernel ──────────────────────────────────────────────────────────────────────
if [ -n "$KERNEL" ]; then
    echo "==> registering the Jupyter kernel '$KERNEL'"
    KENV=()
    [ "$eos_ok" = 1 ] && KENV=(--env MUSIC_EOS_TABLE "$EOS_TABLE")
    "$VPY" -m ipykernel install --user --name "$KERNEL" \
        --display-name "Python ($KERNEL)" ${KENV[@]+"${KENV[@]}"}
fi

# ── check ───────────────────────────────────────────────────────────────────────────────
echo "==> checking the environment"
CHECK_ARGS=()
[ -n "$CHECK_DIR" ] && CHECK_ARGS=("$CHECK_DIR")
[ "$GCS" = 1 ] || [ "$PELICAN" = 1 ] && CHECK_ARGS+=(--remote-test)   # one public read each
( [ "$eos_ok" = 1 ] && export MUSIC_EOS_TABLE="$EOS_TABLE"; "$VPY" "$HERE/check_env.py" ${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"} )

cat <<EOF

Done. Use it with

    source $VENV/bin/activate
    cd $PYJETSCAPE/example/prod_AuAu_0_10_jet
    python run_h5toROOT.py out -j 4                  # hadron files -> ROOT (uproot)
    python $HERE/check_env.py out     # open a production directory
    jupyter notebook jet_wake.ipynb
EOF
[ "$VIZ" = 1 ] && cat <<EOF
    cd $CONTRIBS/Visualization
    python hydro_jet_particles_pyvista.py --file <out>/<stem>.h5 --movie jet_particles.mp4
    python wake_pyvista.py --file <out>/<stem>.h5 --movie wake.mp4
EOF
true
