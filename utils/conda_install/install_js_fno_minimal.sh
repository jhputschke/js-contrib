#!/usr/bin/env bash
# Minimal install script for the js_fno conda environment.
# Only top-level/main packages are listed; conda/pip resolve dependencies.
# CUDA 12.x PyTorch builds are backward-compatible with the CUDA 13 driver.
#
# Usage: bash install_js_fno_minimal.sh [CUDA_VERSION] [CONDA_PREFIX]
#   CUDA_VERSION   pytorch-cuda version to use, "none" for CPU/MPS (Mac Silicon),
#                  or omit to auto-detect from nvcc/nvidia-smi (falls back to "none")
#   CONDA_PREFIX   directory to install Miniconda into if conda is not found (default: $HOME/miniconda3)
#   Examples:
#     bash install_js_fno_minimal.sh                        # auto-detect CUDA
#     bash install_js_fno_minimal.sh 12.1                   # force CUDA 12.1
#     bash install_js_fno_minimal.sh none                   # CPU/MPS — use this on Mac Silicon
#     bash install_js_fno_minimal.sh none /opt/miniconda3   # CPU/MPS, custom conda prefix
#
# Environment variables:
#   JS_FNO_ENV_NAME=<name>   the environment's name (default js_fno)
#   JS_FNO_FORCE=1           replace an existing environment of that name (else: stop)
#   JS_FNO_NO_EDITABLE=1     skip the final `pip install -e` of PyJetscape and FastHydro
set -euo pipefail

ENV_NAME="${JS_FNO_ENV_NAME:-js_fno}"      # JS_FNO_ENV_NAME=<name>: another name
PYTHON_VERSION="3.11"
CONDA_PREFIX="${2:-${HOME}/miniconda3}"

# ---------------------------------------------------------------------------
# CUDA auto-detection: tries nvcc first, then nvidia-smi
# ---------------------------------------------------------------------------
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
        echo "==> Auto-detected CUDA ${CUDA_VERSION}"
    else
        CUDA_VERSION="none"
        echo "==> No CUDA detected — using CPU/MPS mode"
    fi
else
    CUDA_VERSION="$1"
fi

# ---------------------------------------------------------------------------
# Ensure conda is available; offer to install Miniconda if it is not
# ---------------------------------------------------------------------------
if ! command -v conda &>/dev/null; then
    echo "conda not found."
    read -r -p "Install Miniconda to '${CONDA_PREFIX}'? [y/N] " _reply
    if [[ ! "${_reply}" =~ ^[Yy]$ ]]; then
        echo "Aborting — conda is required." >&2
        exit 1
    fi

    # Pick the right installer for the current OS / architecture
    _os="$(uname -s)"
    _arch="$(uname -m)"
    case "${_os}-${_arch}" in
        Linux-x86_64)   _installer="Miniconda3-latest-Linux-x86_64.sh" ;;
        Linux-aarch64)  _installer="Miniconda3-latest-Linux-aarch64.sh" ;;
        Darwin-x86_64)  _installer="Miniconda3-latest-MacOSX-x86_64.sh" ;;
        Darwin-arm64)   _installer="Miniconda3-latest-MacOSX-arm64.sh" ;;
        *)
            echo "Unsupported platform: ${_os}-${_arch}" >&2
            exit 1 ;;
    esac

    _url="https://repo.anaconda.com/miniconda/${_installer}"
    echo "==> Downloading ${_url}"
    curl -fsSL -o "/tmp/${_installer}" "${_url}"

    echo "==> Installing Miniconda to '${CONDA_PREFIX}'"
    bash "/tmp/${_installer}" -b -p "${CONDA_PREFIX}"
    rm "/tmp/${_installer}"

    # Initialise conda for this shell session without requiring a new login
    # shellcheck source=/dev/null
    source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
    echo "==> Miniconda installed. To make conda available in future shells run:"
    echo "    ${CONDA_PREFIX}/bin/conda init $(basename "${SHELL}")"
else
    # Source conda.sh so 'conda activate' works inside the script if needed
    _conda_base="$(conda info --base 2>/dev/null)"
    # shellcheck source=/dev/null
    source "${_conda_base}/etc/profile.d/conda.sh"
fi

# Use mamba if available (faster solver), otherwise fall back to conda
SOLVER="conda"
command -v mamba &>/dev/null && SOLVER="mamba"

# ---------------------------------------------------------------------------
# Never replace an existing environment by accident (conda create -y would)
# ---------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    if [[ "${JS_FNO_FORCE:-0}" != "1" ]]; then
        echo "Environment '${ENV_NAME}' exists already. Remove it first" >&2
        echo "(conda env remove -n ${ENV_NAME}), set JS_FNO_FORCE=1 to replace it, or" >&2
        echo "choose another name with JS_FNO_ENV_NAME=<name>." >&2
        exit 1
    fi
    echo "==> JS_FNO_FORCE=1: replacing the existing environment '${ENV_NAME}'"
fi

echo "==> Creating environment '${ENV_NAME}' with Python ${PYTHON_VERSION}"
${SOLVER} create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# The new environment's own python, by full path: `conda run -n ENV pip` takes the first
# pip on PATH, which is another environment's when a venv is active
ENV_PY="$(conda env list | awk -v n="${ENV_NAME}" '$1 == n {print $NF}')/bin/python"
if [[ ! -x "${ENV_PY}" ]]; then
    echo "Can't find the python of the new environment '${ENV_NAME}' (${ENV_PY})" >&2
    exit 1
fi

echo "==> Installing ROOT (conda-forge) — do this before PyTorch to let the solver run first"
# ROOT >= 6.34 reads the RNTuple files of run_h5toROOT.py.  < 6.38: ROOT 6.38 titles a
# truncated-float TTree leaf with the whole leaf list ('x[n]/f[0,0,12]'), which uproot
# can't parse (UnboundLocalError), so --format ttree --bits-p files become unreadable
# with uproot.  Lift the cap once uproot handles it.
${SOLVER} install -n "${ENV_NAME}" "root>=6.34,<6.38" -c conda-forge -y

if [[ "${CUDA_VERSION}" == "none" ]]; then
    echo "==> Installing PyTorch (CPU/MPS — no CUDA)"
    "${ENV_PY}" -m pip install torch torchvision
else
    # Convert "12.4" -> "cu124" for the PyTorch wheel index
    CUDA_TAG="cu$(echo "${CUDA_VERSION}" | tr -d '.')"
    echo "==> Installing PyTorch with CUDA ${CUDA_VERSION} support (${CUDA_TAG})"
    "${ENV_PY}" -m pip install torch torchvision \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

echo "==> Installing scientific / analysis packages (conda-forge)"
${SOLVER} install -n "${ENV_NAME}" \
    numpy matplotlib scipy h5py seaborn tqdm \
    jupyterlab notebook ipykernel \
    hdf5plugin pyyaml pandas ipywidgets \
    pybind11 pytest networkx vector fastjet \
    -c conda-forge -y

echo "==> Installing pip-only packages"
# With their dependencies (none of them requires torch, so the CUDA build stays):
#   uproot/awkward: awkward-cpp, cramjam, xxhash, fsspec; neuraloperator: tensorly, ...
#   pyvista (+ vtk), imageio, imageio-ffmpeg: contribs/Visualization
"${ENV_PY}" -m pip install \
    "neuraloperator>=2.0" \
    "uproot>=5" \
    "awkward>=2" \
    "pyvista>=0.43" \
    "imageio>=2.9" \
    imageio-ffmpeg

# ---------------------------------------------------------------------------
# The js-contrib Python packages, editable, when the script runs from a checkout
# (pyjetscape_core is built later, with X-SCAPE; `import jetscape` works before,
# with jetscape.HAS_CORE False)
# ---------------------------------------------------------------------------
_JS_CONTRIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "${_JS_CONTRIB}/contribs/PyJetscape/pyproject.toml" && "${JS_FNO_NO_EDITABLE:-0}" != "1" ]]; then
    echo "==> pip install -e PyJetscape and FastHydro from ${_JS_CONTRIB}"
    "${ENV_PY}" -m pip install -e "${_JS_CONTRIB}/contribs/PyJetscape" \
        -e "${_JS_CONTRIB}/contribs/FastHydro"
fi

echo ""
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "Register Jupyter kernel:  ${ENV_PY} -m ipykernel install --user --name ${ENV_NAME}"
