# Technology Stack

**Analysis Date:** 2026-04-15

## Languages

**Primary:**
- Python 3.11+ - Core pipeline, ML checks, batch processing
  - Versions 3.12, 3.13 explicitly tested and supported
  - Minimum: Python 3.11 (verified in `validate.sh`)

**Secondary:**
- Bash - Installation and setup automation (`validate.sh`)
- C++ - Compiled extensions for Hands23 and legacy 100DOH detectors

## Runtime

**Environment:**
- CPython 3.11+ running in isolated virtual environment (`.venv`)
- Virtual environment created and managed by `validate.sh`

**Package Manager:**
- pip - Primary package installation
- Lockfile: Not present (uses requirements.txt and pyproject.toml version constraints)

## Frameworks

**Core:**
- PyTorch 2.x - Deep learning foundation
  - GPU variant: CUDA 12.1 via `https://download.pytorch.org/whl/cu121`
  - CPU variant: `https://download.pytorch.org/whl/cpu` (auto-detected or forced via `FORCE_CPU=1`)
  - Version auto-detected from existing installation or installed fresh

**ML Detection & Computer Vision:**
- YOLOv11 (Ultralytics) - Object detection and pose estimation
  - Used for: Person detection (Phase 1), hand-object interaction, body part visibility (Phase 2)
  - Models: `yolo11s.pt` (default), `yolo11m-pose.pt` (optional, body part visibility)
- SCRFD-2.5GF (InsightFace) - Face detection
  - Used for: Face presence checks (Phase 0 metadata, Phase 1 filtering, Phase 2 validation)
  - Deployed via InsightFace SDK with buffalo_sc weights
- Hands23 Detector (NeurIPS 2023) - Hand-object interaction detection
  - Custom architecture built on Detectron2
  - Used for: Hand visibility, hand-object interaction detection
  - Requires: Detectron2, PyTorch, and compiled C++ extensions
- Grounding DINO - Object grounding for privacy checks
  - Optional, enabled via `--privacy-check` flag
  - HuggingFace variant via `transformers` library

**Vision & Image Processing:**
- OpenCV 4.9.0+ - Frame extraction, image processing, optical flow
  - CUDA-aware build with `cudacodec` module for NVDEC GPU decoding (when available)
  - Falls back to CPU decoding via `cv2.VideoCapture` if NVDEC unavailable
  - Custom cv2.dnn shim in `bachman_cortex/_cv2_dnn_shim.py` for ONNX preprocessing
- Pillow 10.0+ - Image manipulation and format handling

**Deep Learning Infrastructure:**
- ONNX Runtime 1.17.0+ - Model inference acceleration
  - CPU variant for local development
  - GPU variant (`onnxruntime-gpu`) on AWS/cloud deployments
- Detectron2 (Facebook) - Detection framework
  - Installed from GitHub: `git+https://github.com/facebookresearch/detectron2.git`
  - Required for Hands23 detector runtime and model architecture
  - Requires `--no-build-isolation` for compilation

**Utilities & Data Processing:**
- NumPy 1.24.0+ - Array and numerical operations
- Supervision 0.18.0+ - Detection box filtering and visualization utilities
- TQDM 4.65+ - Progress bars for frame extraction and batch processing
- gdown 5.0+ - Google Drive model weight downloads (legacy 100DOH)
- transformers 4.36.0+ - HuggingFace model loading for Grounding DINO
- easydict - Config dictionary access (legacy 100DOH dependency)
- PyYAML - Configuration file parsing
- SciPy - Scientific computing utilities
- Cython - C extension compilation (legacy 100DOH)

**Build & Development:**
- setuptools 68.0+ - Package build and installation
- pytest - Testing framework (referenced in benchmarking tests)

## Key Dependencies

**Critical ML Models:**
- `insightface>=0.7.3` - SCRFD face detector with pre-trained buffalo_sc weights (~50MB)
- `ultralytics>=8.1.0` - YOLOv11 with auto-download from HuggingFace (~90MB for yolo11s)
- `torch` 2.x - PyTorch (1.5-2.5GB depending on variant)
- `torchvision` - Computer vision models and transforms
- `detectron2` (GitHub) - Backbone for Hands23, requires compilation

**Model Weights (Downloaded at setup):**
- SCRFD: ~50MB (via InsightFace auto-download)
- YOLO11s: ~90MB (via Ultralytics auto-download)
- YOLO11m-pose: ~210MB (optional, via Ultralytics auto-download)
- Hands23: ~400MB from `https://fouheylab.eecs.umich.edu/~dandans/projects/hands23/`
- 100DOH (legacy): ~360MB from Google Drive (optional, via `--100doh` flag)
- Grounding DINO: ~2.5GB (if `--privacy-check` enabled)

**Infrastructure:**
- FFmpeg (external) - Video encoding/decoding
  - `ffprobe` required for metadata extraction (Phase 0)
  - Optional `ffmpeg` for HEVC-to-H.264 transcoding via NVENC or libx264

**Optional GPU Acceleration:**
- NVIDIA CUDA 12.1 - GPU compute (auto-detected, optional)
- NVIDIA cuDNN - GPU-accelerated deep learning primitives (installed with PyTorch)
- NVIDIA NVDEC - Hardware video decoding (auto-detected, fallback to CPU decode)
- NVIDIA NVENC - Hardware video encoding for transcoding (auto-detected, fallback to libx264 -crf 0)

## Configuration

**Environment:**
- Configured via CLI arguments passed to `hl-validate` command
- No environment variables for secrets (all configuration is CLI or config objects)
- Python version auto-detection in `validate.sh` (checks 3.12, 3.11, 3.13 in order)

**Build:**
- `pyproject.toml` - Project metadata and dependencies
  - Entry point: `hl-validate = bachman_cortex.run_batch:main`
  - Dependencies listed with version constraints but PyTorch/Detectron2 handled separately
- `bachman_cortex/requirements.txt` - Manual installation reference (not used by validate.sh)
- `validate.sh` - Orchestrates full setup in order: Python → PyTorch → Detectron2 → ONNX Runtime → Package install → Model downloads

**Pipeline Configuration:**
- `bachman_cortex/pipeline.py:PipelineConfig` - Dataclass with 30+ configurable thresholds
  - Face confidence: 0.8 (default)
  - Hand confidence: 0.7 (default)
  - Hand pass rate: 0.80 (both hands) / 0.90 (single hand)
  - Motion shakiness threshold: 0.50
  - Brightness stability std: 60.0
  - Overridable via CLI args to `hl-validate`

## Platform Requirements

**Development:**
- Python 3.11+ (3.12 recommended)
- FFmpeg with ffprobe
- Git (for installing detectron2 and cloning model repos)
- 2GB+ disk space (model weights + venv)
- C++ compiler for Detectron2/Hands23 compilation:
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Ubuntu/Debian: `build-essential` package

**GPU (Optional):**
- NVIDIA GPU with CUDA Compute Capability 3.5+ (Kepler generation or newer)
- NVIDIA driver supporting CUDA 12.1+
- VRAM: 2GB minimum per worker (multi-worker auto-scales based on available VRAM)

**Platforms Tested:**
- macOS (Apple Silicon and Intel)
- Ubuntu/Debian Linux
- Windows: Not supported (Detectron2 lacks Windows support; WSL2 alternative provided)

**Production:**
- AWS EC2 with GPU (g4dn.xlarge or similar for NVENC/NVDEC)
- Cloud container images with pre-compiled dependencies
- Local batch processing on CPU (slower but functional)

---

*Stack analysis: 2026-04-15*
