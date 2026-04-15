#!/usr/bin/env bash
# Build OpenCV 4.10.0 with CUDA for this project's venv.
#
# Why: the pipeline's motion_analysis.py path `cv2.cuda.SparsePyrLKOpticalFlow`
# is dead because PyPI opencv-python ships without CUDA. See
# OPTIMIZATION_PLAN_V3.md §2 / P2.1.
#
# Scope: builds the cudaoptflow / cudaarithm / cudaimgproc / cudawarping
# modules that back `cv2.cuda.SparsePyrLKOpticalFlow`. Does NOT build:
#   - cv2.dnn CUDA backend (would need cuDNN — the project uses ORT/PyTorch,
#     not cv2.dnn, so it's unused weight).
#   - cv2.cudacodec NVDEC decode (would need NVIDIA Video Codec SDK, which
#     is behind a developer-portal login — nice-to-have, not required).
# Re-enable those with WITH_CUDNN/WITH_NVCUVID plus manual dep install.
#
# Requirements (verified manually — this script does NOT auto-install):
#   - sudo access (apt)
#   - NVIDIA driver present (nvidia-smi works) — confirmed 580.126.09 CUDA 13
#   - ~5 GB free disk, ~1 hr build time on RTX 3060 host
#
# Usage:
#   ./scripts/install_opencv_cuda.sh
#
# The venv-local opencv-python/opencv-contrib-python wheels will be
# uninstalled and replaced with a CUDA-enabled cv2 module.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
BUILD_DIR="${BUILD_DIR:-$HOME/build/opencv-cuda}"
OPENCV_VERSION="${OPENCV_VERSION:-4.10.0}"
CUDA_ARCH="${CUDA_ARCH:-8.6}"  # RTX 3060 = sm_86
# Path to extracted NVIDIA Video Codec SDK (enables cv2.cudacodec.VideoReader
# NVDEC decode). Leave unset to build without NVDEC. Download from
# https://developer.nvidia.com/video-codec-sdk-archive (requires dev login).
NVCODEC_SDK="${NVCODEC_SDK:-}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: venv not found at $VENV_DIR — run ./validate.sh --setup-only first"
    exit 1
fi

echo "==> Step 1/5: apt install system deps (requires sudo password)"
sudo apt update
sudo apt install -y cmake ninja-build build-essential git pkg-config \
    nvidia-cuda-toolkit libgtk-3-dev libavcodec-dev libavformat-dev \
    libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev \
    libpng-dev libtiff-dev gfortran python3-dev

# Install NVIDIA Video Codec SDK headers + stub libs if provided. These
# aren't on apt; user must download from developer.nvidia.com.
if [ -n "$NVCODEC_SDK" ]; then
    if [ ! -f "$NVCODEC_SDK/Interface/nvcuvid.h" ]; then
        echo "ERROR: NVCODEC_SDK=$NVCODEC_SDK — Interface/nvcuvid.h not found"
        exit 1
    fi
    echo "==> Step 1b/5: install Video Codec SDK headers + stub libs"
    sudo install -m 0644 "$NVCODEC_SDK/Interface/nvcuvid.h" /usr/include/
    sudo install -m 0644 "$NVCODEC_SDK/Interface/cuviddec.h" /usr/include/
    sudo install -m 0644 "$NVCODEC_SDK/Interface/nvEncodeAPI.h" /usr/include/
    sudo install -m 0755 "$NVCODEC_SDK/Lib/linux/stubs/x86_64/libnvcuvid.so" \
        /usr/lib/x86_64-linux-gnu/
    sudo install -m 0755 "$NVCODEC_SDK/Lib/linux/stubs/x86_64/libnvidia-encode.so" \
        /usr/lib/x86_64-linux-gnu/
    sudo ldconfig
    echo "  SDK files installed."
fi

echo "==> Step 2/5: clone OpenCV ${OPENCV_VERSION} + contrib"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
if [ ! -d opencv ]; then
    git clone --branch "$OPENCV_VERSION" --depth 1 \
        https://github.com/opencv/opencv.git
fi
if [ ! -d opencv_contrib ]; then
    git clone --branch "$OPENCV_VERSION" --depth 1 \
        https://github.com/opencv/opencv_contrib.git
fi

echo "==> Step 3/5: configure with CUDA (no cuDNN; NVCUVID if SDK provided)"
if [ -n "$NVCODEC_SDK" ]; then
    CUDA_NVCUVID="ON"
    CUDA_NVCUVENC="ON"
else
    CUDA_NVCUVID="OFF"
    CUDA_NVCUVENC="OFF"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PY_EXECUTABLE="$(which python)"
PY_INCLUDE="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')"
PY_SITEPACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
NUMPY_INCLUDE="$(python -c 'import numpy; print(numpy.get_include())')"

# CUDA 12.0's nvcc caps supported host compiler at gcc 12. Ubuntu 24.04
# defaults to gcc 13, so we must pin the CUDA host compiler to gcc-12
# (installed as a transitive dep of nvidia-cuda-toolkit).
if ! command -v gcc-12 >/dev/null || ! command -v g++-12 >/dev/null; then
    echo "ERROR: gcc-12 / g++-12 not found. Install: sudo apt install gcc-12 g++-12"
    exit 1
fi
CUDA_HOST_GCC="$(command -v gcc-12)"
CUDA_HOST_GXX="$(command -v g++-12)"

mkdir -p "$BUILD_DIR/opencv/build"
cd "$BUILD_DIR/opencv/build"
# Wipe any stale cmake cache from a prior failed configure so we don't
# inherit stale feature detections.
rm -f CMakeCache.txt
rm -rf CMakeFiles
cmake -G Ninja \
    -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="$BUILD_DIR/install" \
    -D OPENCV_EXTRA_MODULES_PATH="$BUILD_DIR/opencv_contrib/modules" \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=OFF \
    -D WITH_NVCUVID="$CUDA_NVCUVID" \
    -D WITH_NVCUVENC="$CUDA_NVCUVENC" \
    -D WITH_CUBLAS=ON \
    -D OPENCV_DNN_CUDA=OFF \
    -D BUILD_opencv_dnn=OFF \
    -D BUILD_opencv_mcc=OFF \
    -D CUDA_ARCH_BIN="$CUDA_ARCH" \
    -D CUDA_ARCH_PTX="" \
    -D CUDA_FAST_MATH=ON \
    -D CUDA_HOST_COMPILER="$CUDA_HOST_GCC" \
    -D CMAKE_CUDA_HOST_COMPILER="$CUDA_HOST_GCC" \
    -D CMAKE_C_COMPILER="$CUDA_HOST_GCC" \
    -D CMAKE_CXX_COMPILER="$CUDA_HOST_GXX" \
    -D BUILD_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE="$PY_EXECUTABLE" \
    -D PYTHON3_INCLUDE_DIR="$PY_INCLUDE" \
    -D PYTHON3_NUMPY_INCLUDE_DIRS="$NUMPY_INCLUDE" \
    -D PYTHON3_PACKAGES_PATH="$PY_SITEPACKAGES" \
    -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=OFF \
    ..

echo "==> Step 4/5: build (~45–60 min on RTX 3060 host)"
ninja -j"$(nproc)"

echo "==> Step 5/5: install into venv"
ninja install
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true

python - <<'PY'
import cv2
print("cv2 version:", cv2.__version__)
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
assert cv2.cuda.getCudaEnabledDeviceCount() > 0, "CUDA build failed — 0 devices"
# Smoke test cv2.cuda.SparsePyrLKOpticalFlow (the motion_analysis.py target)
try:
    cv2.cuda.SparsePyrLKOpticalFlow.create()
    print("SparsePyrLKOpticalFlow: OK")
except Exception as e:
    raise SystemExit(f"SparsePyrLKOpticalFlow not available: {e}")
# NVDEC availability (optional — only if WITH_NVCUVID was on).
if hasattr(cv2, "cudacodec"):
    print("cudacodec: available (NVDEC enabled)")
else:
    print("cudacodec: not built (NVCUVID=OFF)")
print("SUCCESS: CUDA OpenCV installed")
PY
