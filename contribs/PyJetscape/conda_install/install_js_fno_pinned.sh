#!/usr/bin/env bash
# Pinned install script for the js_fno conda environment.
# All main packages are fixed to specific versions for reproducibility.
# CUDA 12.x PyTorch builds are backward-compatible with the CUDA 13 driver.
#
# Usage: bash install_js_fno_pinned.sh [CUDA_VERSION] [CONDA_PREFIX]
#   CUDA_VERSION   pytorch-cuda version to use, "none" for CPU/MPS (Mac Silicon),
#                  or omit to auto-detect from nvcc/nvidia-smi (falls back to "none")
#   CONDA_PREFIX   directory to install Miniconda into if conda is not found (default: $HOME/miniconda3)
#   Examples:
#     bash install_js_fno_pinned.sh                        # auto-detect CUDA
#     bash install_js_fno_pinned.sh 12.1                   # force CUDA 12.1
#     bash install_js_fno_pinned.sh none                   # CPU/MPS — use this on Mac Silicon
#     bash install_js_fno_pinned.sh none /opt/miniconda3   # CPU/MPS, custom conda prefix
set -euo pipefail

ENV_NAME="js_fno"
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

SOLVER="conda"
command -v mamba &>/dev/null && SOLVER="mamba"

echo "==> Creating environment '${ENV_NAME}' with pinned Python"
${SOLVER} create -n "${ENV_NAME}" python=3.11.9 -y

echo "==> Installing ROOT (conda-forge)"
${SOLVER} install -n "${ENV_NAME}" \
    root=6.32.2 \
    -c conda-forge -y

if [[ "${CUDA_VERSION}" == "none" ]]; then
    echo "==> Installing PyTorch 2.4.1 (CPU/MPS — no CUDA)"
    conda run -n "${ENV_NAME}" pip install \
        "torch==2.4.1" \
        "torchvision==0.19.1"
else
    # Convert "12.4" -> "cu124" for the PyTorch wheel index
    CUDA_TAG="cu$(echo "${CUDA_VERSION}" | tr -d '.')"
    echo "==> Installing PyTorch 2.4.1 with CUDA ${CUDA_VERSION} support (${CUDA_TAG})"
    conda run -n "${ENV_NAME}" pip install \
        "torch==2.4.1" \
        "torchvision==0.19.1" \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

echo "==> Installing scientific / analysis packages (conda-forge)"
${SOLVER} install -n "${ENV_NAME}" \
    numpy=1.26.4 \
    matplotlib=3.9.2 \
    scipy=1.14.1 \
    h5py=3.11.0 \
    seaborn=0.13.2 \
    tqdm=4.66.5 \
    jupyterlab=4.2.5 \
    ipykernel=6.29.5 \
    -c conda-forge -y

echo "==> Installing pinned pip packages"
conda run -n "${ENV_NAME}" pip install \
    "neuraloperator==2.0.0" \
    "uproot==5.3.3" \
    "awkward==2.6.5"

echo ""
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "Register Jupyter kernel:  conda run -n ${ENV_NAME} python -m ipykernel install --user --name ${ENV_NAME}"
