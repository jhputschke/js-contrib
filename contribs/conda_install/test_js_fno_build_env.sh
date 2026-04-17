#!/usr/bin/env bash
# Dry-run test for install_js_fno_build_minimal.sh.
# Checks every package for availability without creating an environment
# or installing anything.  Prints a PASS/FAIL summary at the end.
#
# Usage: bash test_js_fno_build_env.sh [CUDA_VERSION]
#   CUDA_VERSION   same meaning as in install_js_fno_build_minimal.sh;
#                  omit to auto-detect (falls back to "none")
set -uo pipefail   # no -e so we can collect failures instead of aborting

PASS=0
FAIL=0
WARN=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
_red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
_yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }

_ok()   { _green  "  [PASS] $*"; ((PASS++));  }
_fail() { _red    "  [FAIL] $*"; ((FAIL++));  }
_warn() { _yellow "  [WARN] $*"; ((WARN++)); }

# Check a conda-forge package.
# If conda is in PATH, use `conda search`; otherwise fall back to the
# anaconda.org REST API via curl so the check works before conda is installed.
# Uses explicit HTTP status check to avoid false failures from rate-limit redirects.
_check_conda() {
    local pkg="$1"
    if command -v conda &>/dev/null; then
        if conda search -c conda-forge --override-channels "${pkg}" &>/dev/null 2>&1; then
            _ok   "conda-forge: ${pkg}"
        else
            _fail "conda-forge: ${pkg} — not found"
        fi
    else
        local _status
        _status=$(curl -sSL --max-time 10 -o /dev/null \
            -w "%{http_code}" \
            "https://api.anaconda.org/package/conda-forge/${pkg}" 2>/dev/null)
        if [[ "${_status}" == "200" ]]; then
            _ok   "conda-forge: ${pkg} (via anaconda.org API)"
        else
            _fail "conda-forge: ${pkg} — not found or API unavailable (HTTP ${_status})"
        fi
    fi
}

# Check a pip package.
# Tries `pip index versions` first; falls back to the PyPI JSON API via curl
# so the check works even with older pip or when pip index is unavailable.
_check_pip() {
    local pkg="$1"
    if command -v pip &>/dev/null && pip index versions "${pkg}" &>/dev/null 2>&1; then
        _ok   "PyPI: ${pkg}"
    else
        local _status
        _status=$(curl -sSL --max-time 10 -o /dev/null \
            -w "%{http_code}" \
            "https://pypi.org/pypi/${pkg}/json" 2>/dev/null)
        if [[ "${_status}" == "200" ]]; then
            _ok   "PyPI: ${pkg} (via pypi.org API)"
        else
            _fail "PyPI: ${pkg} — not found (HTTP ${_status})"
        fi
    fi
}

# ---------------------------------------------------------------------------
# 1. CUDA detection
# ---------------------------------------------------------------------------
echo ""
echo "=== 1. CUDA detection ==="

_detect_cuda() {
    if command -v nvcc &>/dev/null; then
        nvcc --version 2>/dev/null \
            | grep -o 'release [0-9]*\.[0-9]*' \
            | grep -o '[0-9]*\.[0-9]*'
    elif command -v nvidia-smi &>/dev/null; then
        nvidia-smi 2>/dev/null \
            | grep 'CUDA Version' \
            | grep -o '[0-9]*\.[0-9]*' \
            | head -1
    fi
}

if [[ -z "${1:-}" ]]; then
    _detected="$(_detect_cuda)"
    if [[ -n "${_detected}" ]]; then
        CUDA_VERSION="${_detected}"
        _ok "Auto-detected CUDA ${CUDA_VERSION}"
    else
        CUDA_VERSION="none"
        _warn "No CUDA detected — will use CPU/MPS mode"
    fi
else
    CUDA_VERSION="$1"
    _ok "Using provided CUDA_VERSION=${CUDA_VERSION}"
fi

if [[ "${CUDA_VERSION}" != "none" ]]; then
    CUDA_TAG="cu$(echo "${CUDA_VERSION}" | tr -d '.')"
    TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
    echo "     PyTorch index URL: ${TORCH_INDEX}"
fi

# ---------------------------------------------------------------------------
# 2. Conda availability
# ---------------------------------------------------------------------------
echo ""
echo "=== 2. Conda availability ==="

if command -v conda &>/dev/null; then
    _ok "conda found: $(conda --version 2>&1)"
else
    _warn "conda not found — install_js_fno_build_minimal.sh will offer to install Miniconda"
fi

if command -v mamba &>/dev/null; then
    _ok "mamba found (faster solver): $(mamba --version 2>&1 | head -1)"
else
    _warn "mamba not found — will fall back to conda solver (slower)"
fi

# ---------------------------------------------------------------------------
# 3. JETSCAPE C++ build dependencies (conda-forge)
# ---------------------------------------------------------------------------
echo ""
echo "=== 3. JETSCAPE C++ build dependencies (conda-forge) ==="

for pkg in cmake make compilers boost-cpp zlib hdf5 pythia8 hepmc3 fastjet gsl openmpi; do
    _check_conda "${pkg}"
done

# ---------------------------------------------------------------------------
# 4. ROOT (conda-forge)
# ---------------------------------------------------------------------------
echo ""
echo "=== 4. ROOT (conda-forge) ==="
_check_conda root

# ---------------------------------------------------------------------------
# 5. PyTorch wheels (pip)
# ---------------------------------------------------------------------------
echo ""
echo "=== 5. PyTorch wheels (pip) ==="

if [[ "${CUDA_VERSION}" == "none" ]]; then
    echo "     Checking PyPI (CPU/MPS build) ..."
    _check_pip torch
    _check_pip torchvision
else
    echo "     Checking download.pytorch.org/whl/${CUDA_TAG} ..."
    # Use pip index with --index-url to verify CUDA-specific wheels exist
    if pip index versions torch \
            --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
            &>/dev/null 2>&1; then
        _ok   "PyTorch wheel index reachable for ${CUDA_TAG}"
    else
        _fail "PyTorch wheel index unreachable or no torch wheels for ${CUDA_TAG}"
    fi
fi

# ---------------------------------------------------------------------------
# 6. Python analysis packages (conda-forge)
# ---------------------------------------------------------------------------
echo ""
echo "=== 6. Python analysis packages (conda-forge) ==="

for pkg in numpy matplotlib scipy h5py seaborn tqdm jupyterlab ipykernel; do
    _check_conda "${pkg}"
done

# ---------------------------------------------------------------------------
# 7. pip-only packages
# ---------------------------------------------------------------------------
echo ""
echo "=== 7. pip-only packages ==="

_check_pip neuraloperator
_check_pip uproot
_check_pip awkward

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=============================="
echo " Results: ${PASS} passed  |  ${FAIL} failed  |  ${WARN} warnings"
echo "=============================="

if [[ ${FAIL} -gt 0 ]]; then
    _red  "One or more checks FAILED — review the output above before running the install script."
    exit 1
else
    _green "All checks passed — safe to run install_js_fno_build_minimal.sh."
fi
