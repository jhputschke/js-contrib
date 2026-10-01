#!/usr/bin/env bash
# Full pinned install script for the js_fno conda environment.
# Installs both the JETSCAPE C++ build dependencies (Boost, Pythia8, HDF5, …)
# and the Python/ML stack (PyTorch via pip, ROOT, neuraloperator, uproot, …).
# All main packages are fixed to specific versions for reproducibility.
#
# Usage: bash install_js_fno_build_pinned.sh [CUDA_VERSION] [CONDA_PREFIX]
#   CUDA_VERSION   pytorch wheel CUDA tag, "none" for CPU/MPS (Mac Silicon),
#                  or omit to auto-detect from nvcc/nvidia-smi (falls back to "none")
#   CONDA_PREFIX   directory to install Miniconda into if conda is not found (default: $HOME/miniconda3)
#   Examples:
#     bash install_js_fno_build_pinned.sh                        # auto-detect CUDA
#     bash install_js_fno_build_pinned.sh 12.1                   # force CUDA 12.1
#     bash install_js_fno_build_pinned.sh none                   # CPU/MPS — Mac Silicon
#     bash install_js_fno_build_pinned.sh none /opt/miniconda3   # custom conda prefix
#
# Environment variables:
#   JS_FNO_ENV_NAME=<name>   the environment's name (default js_fno)
#   JS_FNO_FORCE=1           replace an existing environment of that name (else: stop)
#   JS_FNO_NO_EDITABLE=1     skip the final `pip install -e` of PyJetscape and FastHydro
#   JS_FNO_NO_MLX=1          skip MLX (installed on Apple Silicon, Darwin-arm64, only)
set -euo pipefail

ENV_NAME="${JS_FNO_ENV_NAME:-js_fno}"      # JS_FNO_ENV_NAME=<name>: another name
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

# ---------------------------------------------------------------------------
# Create environment
# ---------------------------------------------------------------------------
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

echo "==> Creating environment '${ENV_NAME}' with pinned Python"
${SOLVER} create -n "${ENV_NAME}" python=3.11.9 -y

# The new environment's own python, by full path: `conda run -n ENV pip` takes the first
# pip on PATH, which is another environment's when a venv is active
ENV_PY="$(conda env list | awk -v n="${ENV_NAME}" '$1 == n {print $NF}')/bin/python"
if [[ ! -x "${ENV_PY}" ]]; then
    echo "Can't find the python of the new environment '${ENV_NAME}' (${ENV_PY})" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# JETSCAPE C++ build dependencies (conda-forge)
# CMakeLists.txt required: Boost, ZLIB, Pythia8, HDF5
# CMakeLists.txt optional: HepMC3, FastJet, GSL, OpenMPI, ROOT, cmake
# fastjet is scikit-hep's conda package: the FastJet C++ library and its Python API
# (example/analysis/jet_fastjet*.ipynb)
# ---------------------------------------------------------------------------
echo "==> Installing JETSCAPE C++ build dependencies (conda-forge)"
${SOLVER} install -n "${ENV_NAME}" \
    cmake=3.29.3 \
    make=4.3 \
    compilers \
    boost-cpp=1.85.0 \
    zlib=1.3.1 \
    hdf5=1.14.3 \
    pythia8=8.312 \
    hepmc3=3.2.7 \
    fastjet=3.5.0.1 \
    gsl=2.7 \
    openmpi=4.1.6 \
    -c conda-forge -y

# ---------------------------------------------------------------------------
# ROOT — install before PyTorch so the solver sees all constraints at once
# ---------------------------------------------------------------------------
echo "==> Installing ROOT (conda-forge)"
${SOLVER} install -n "${ENV_NAME}" root=6.32.2 -c conda-forge -y

# ---------------------------------------------------------------------------
# PyTorch (pip — conda channel no longer officially supported)
# ---------------------------------------------------------------------------
if [[ "${CUDA_VERSION}" == "none" ]]; then
    echo "==> Installing PyTorch 2.4.1 (CPU/MPS — no CUDA)"
    "${ENV_PY}" -m pip install \
        "torch==2.4.1" \
        "torchvision==0.19.1"
else
    CUDA_TAG="cu$(echo "${CUDA_VERSION}" | tr -d '.')"
    echo "==> Installing PyTorch 2.4.1 with CUDA ${CUDA_VERSION} support (${CUDA_TAG})"
    "${ENV_PY}" -m pip install \
        "torch==2.4.1" \
        "torchvision==0.19.1" \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

# ---------------------------------------------------------------------------
# Python analysis / ML packages (conda-forge)
# ---------------------------------------------------------------------------
echo "==> Installing Python analysis packages (conda-forge)"
${SOLVER} install -n "${ENV_NAME}" \
    numpy=1.26.4 \
    matplotlib=3.9.2 \
    scipy=1.14.1 \
    h5py=3.11.0 \
    seaborn=0.13.2 \
    tqdm=4.66.5 \
    jupyterlab=4.2.5 \
    ipykernel=6.29.5 \
    metakernel=0.30.2 \
    hdf5plugin=5.0.0 \
    pyyaml=6.0.2 \
    pandas=2.2.3 \
    ipywidgets=8.1.5 \
    notebook=7.2.2 \
    pybind11=2.13.6 \
    pytest=8.3.3 \
    networkx=3.3 \
    -c conda-forge -y

# ---------------------------------------------------------------------------
# pip-only packages
# ---------------------------------------------------------------------------
echo "==> Installing pip packages"
"${ENV_PY}" -m pip install \
    "neuraloperator==2.0.0" \
    "uproot==5.3.3" \
    "awkward==2.6.5" \
    "vector==1.5.1" \
    "pyvista==0.46.3" \
    "vtk==9.5.0" \
    "imageio==2.37.0" \
    "imageio-ffmpeg==0.6.0"

# MLX (Apple's array framework, Metal) on Apple Silicon only; wheels need macOS >= 14,
# so a failed install warns instead of stopping the script.  It doesn't require torch.
if [[ "$(uname -s)-$(uname -m)" == "Darwin-arm64" && "${JS_FNO_NO_MLX:-0}" != "1" ]]; then
    echo "==> Installing MLX (Apple Silicon)"
    "${ENV_PY}" -m pip install "mlx==0.32.3" \
        || echo "WARNING: MLX install failed (macOS >= 14 needed); the env works without it" >&2
fi

# ---------------------------------------------------------------------------
# The js-contrib Python packages, editable, when the script runs from a checkout
# (pyjetscape_core is built later, with X-SCAPE; `import jetscape` works before,
# with jetscape.HAS_CORE False)
# ---------------------------------------------------------------------------
_JS_CONTRIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -f "${_JS_CONTRIB}/contribs/PyJetscape/pyproject.toml" && "${JS_FNO_NO_EDITABLE:-0}" != "1" ]]; then
    echo "==> pip install -e PyJetscape and FastHydro from ${_JS_CONTRIB}"
    "${ENV_PY}" -m pip install -e "${_JS_CONTRIB}/contribs/PyJetscape" \
        -e "${_JS_CONTRIB}/contribs/FastHydro"
fi

echo ""
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "Register Jupyter kernel:  ${ENV_PY} -m ipykernel install --user --name ${ENV_NAME}"
