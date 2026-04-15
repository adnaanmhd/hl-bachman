# External Integrations

**Analysis Date:** 2026-04-15

## APIs & External Services

**Model Download Sources:**
- PyTorch Official Index - GPU/CPU wheel downloads
  - CUDA 12.1: `https://download.pytorch.org/whl/cu121`
  - CPU: `https://download.pytorch.org/whl/cpu`
  - Detection: Auto-selected in `validate.sh` based on GPU availability

- Ultralytics HuggingFace Hub - YOLOv11 model auto-download
  - Models: `yolo11s.pt` (default), `yolo11m-pose.pt` (optional)
  - Auto-downloaded on first use by `from ultralytics import YOLO`
  - Location: `~/.yolov5` or similar cache directory

- Hands23 GitHub Repository & Direct URL
  - Repository: `https://github.com/EvaCheng-cty/hands23_detector.git`
  - Weights: `https://fouheylab.eecs.umich.edu/~dandans/projects/hands23/model_weights/model_hands23.pth` (~400MB)
  - Download method: wget/curl with `--no-check-certificate` for SSL issues, fallback to Python urllib
  - Storage: `bachman_cortex/models/weights/hands23_detector/`
  - Implemented in: `bachman_cortex/models/download_models.py:download_hands23()`

- Facebook Detectron2 GitHub
  - Repository: `git+https://github.com/facebookresearch/detectron2.git`
  - Installation method: Git clone + pip install with `--no-build-isolation`
  - Requires: C++ compiler for source compilation
  - Installed by: `validate.sh` (step 5)

- Legacy 100DOH Hand-Object Detector (optional, `--100doh` flag)
  - Repository: `https://github.com/ddshan/hand_object_detector.git`
  - Weights: Google Drive via `gdown` library
  - Download ID: `1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE`
  - Size: ~360MB (100K+ego model)
  - Storage: `bachman_cortex/models/weights/hand_object_detector/`
  - Implemented in: `bachman_cortex/models/download_models.py:download_100doh()`

**InsightFace API (Face Detection):**
- SDK: `insightface>=0.7.3`
- Model: buffalo_sc (SCRFD-2.5GF)
- Auto-downloads on first use to `bachman_cortex/models/weights/insightface/`
- Initialized in: `bachman_cortex/models/download_models.py:download_scrfd()`
  - Uses CPU execution provider for inference
  - Detection size: 640x640

**HuggingFace Models (Grounding DINO)**
- Framework: `transformers>=4.36.0`
- Model: Grounding DINO (loaded dynamically when `--privacy-check` enabled)
- Auto-downloads from HuggingFace Hub on first use
- Size: ~2.5GB
- No explicit integration code; used via `transformers.AutoModel` pattern

## Data Storage

**Databases:**
- None - Pipeline produces report outputs only (no persistent data store)

**File Storage:**
- Local filesystem only
  - Input: `.mp4` video files (configurable via CLI)
  - Output directory: `bachman_cortex/results/` (default, configurable via `--output`)
  - Model weights: `bachman_cortex/models/weights/` (auto-downloaded)
  - Preprocessed cache: `bachman_cortex/results/.preprocessed/` (temp HEVC transcodes, deleted after pipeline)

**Caching:**
- Frame memory cache within single pipeline run (Phase 1-2)
  - Stored in `phase1_cache` dict in `bachman_cortex/pipeline.py:run()`
  - Freed after Phase 2 to reduce memory footprint
- Motion analyzer results cached and reused for Phase 2 motion checks
- Model weights cached on disk (`~/.yolov5/`, InsightFace local dir, etc.)
- No HTTP cache or distributed caching

## Authentication & Identity

**Auth Provider:**
- None - Fully self-contained, no authentication required
- No API keys or credentials needed
- FFmpeg/git required as system executables (not authenticated)

## Monitoring & Observability

**Error Tracking:**
- None - No error reporting service integration

**Logs:**
- Standard output (stdout)
  - Progress: `==> Setting up...` messages to console
  - Model downloads: Download progress printed to stdout
  - Pipeline execution: Frame extraction and check progress via TQDM progress bars
- Exception handling: Stack traces printed to stderr
- Per-video JSON results: Machine-readable output to `{video_name}.json`
- Per-video Markdown report: Human-readable timeline to `{video_name}.md` (written by `bachman_cortex/reporting.py`)
- Batch summary: `batch_report.md` and `batch_results.json` (summary across all videos)

**Metrics Tracked:**
- Per-check results with pass/fail and confidence scores
- Per-segment validation results
- Yield calculation: usable duration / total duration
- Phase durations and frame processing rates (from benchmarks in `tests/`)

## CI/CD & Deployment

**Hosting:**
- Local machine (development) or EC2 (cloud)
- No hosted inference; all models run locally
- Multi-worker support for parallel video processing via `multiprocessing.Pool`
  - Worker count auto-detected based on CPU cores and available VRAM
  - Configurable via `--workers N` flag in `hl-validate`

**CI Pipeline:**
- None integrated - Project structure supports external CI via `validate.sh --setup-only` + `hl-validate ...`
- All setup automated in single script for easy integration

## Environment Configuration

**Required env vars:**
- None required (all configuration via CLI arguments)

**Optional env vars:**
- `FORCE_CPU=1` - Force CPU-only PyTorch even if GPU detected (useful for testing CPU path)

**Example environment:**
```bash
# GPU-enabled (auto-detected)
./validate.sh /path/to/videos/

# Force CPU
FORCE_CPU=1 ./validate.sh /path/to/videos/

# Setup only (no pipeline run)
./validate.sh --setup-only
```

**Secrets location:**
- No secrets stored or managed by pipeline
- Credentials not used

## Webhooks & Callbacks

**Incoming:**
- None - Pipeline is request-response only (CLI-driven)

**Outgoing:**
- None - Pipeline does not make outbound requests beyond model downloads
- Results written to local files only

## Model Download Pipeline

**Download Strategy:**
- Single-pass automatic download on first run via `validate.sh`
- All downloads triggered by `bachman_cortex/models/download_models.py`
- Total size: ~1.5-2GB (SCRFD + YOLO11s + Hands23 + dependencies)
- Grounding DINO (~2.5GB) downloaded only if `--privacy-check` enabled
- Legacy 100DOH (~360MB) downloaded only if `--100doh` flag used

**Download Sequencing:**
1. SCRFD (InsightFace) - ~50MB
2. YOLO11s (Ultralytics) - ~90MB
3. Hands23 (GitHub + direct URL) - ~400MB + repo clone
4. Detectron2 compilation (if Hands23 not already installed)
5. 100DOH (optional, legacy) - ~360MB + C++ compilation

**Failure Handling:**
- Hands23 download uses fallback mechanisms: wget → curl → urllib with SSL verification disabled
- Google Drive downloads retry via gdown library
- Compilation failures log output and suggest missing build tools (C++ compiler)

## External System Dependencies

**System Executables (required):**
- `python3.11+` - Installed and verified by `validate.sh`
- `ffprobe` - FFmpeg utility for video metadata extraction (Phase 0)
  - Checked in `validate.sh` step 1
  - Error if missing: Provides install instructions for macOS and Ubuntu

- `git` - Required for Detectron2 and Hands23 repo cloning
  - Checked in `validate.sh` step 1
  - Error if missing: Provides install instructions

- `nvidia-smi` (optional) - NVIDIA GPU detection
  - If present and working: Installs GPU-enabled PyTorch + ONNX Runtime
  - If missing: Silently falls back to CPU variants
  - GPU detection in `validate.sh` and `run_batch.py:_auto_detect_workers()`

**Optional System Packages:**
- `libcuda.so` - NVIDIA CUDA runtime (auto-detected by PyTorch)
- `libnvenc.so` - NVIDIA NVENC codec (optional, for hardware video encoding)
- `libnvdec.so` - NVIDIA NVDEC codec (optional, for hardware video decoding in OpenCV)

---

*Integration audit: 2026-04-15*
