# Video Validation Pipeline: Research, Model Selection & Implementation Context

**Date:** 2026-03-30 (updated 2026-04-14)
**Project:** Bachman — Egocentric Video Validation & Submission Tool
**Author:** Adnaan (PM, Humyn Labs) + Claude Code

---

## 1. Problem & Purpose

Humyn Labs is building an egocentric video dataset for **training autonomous humanoid robots**. Videos are captured by a distributed workforce wearing head-mounted cameras while performing agricultural, commercial, and residential tasks. The validation pipeline checks video quality across 20 criteria before data enters the training pipeline.

**Scale:** ~5,000 videos/day, each ~500MB, 5+ minutes, 1080p, 30FPS.
**Target:** <5 min total processing per video.

The pipeline validates 5 categories of checks across 21 total criteria:
1. **Video Metadata** (6 checks) — format, encoding, resolution, frame rate, duration, orientation. Acts as a gate: failure skips all other checks.
2. **Frame-Level Quality** (3 checks) — average brightness, brightness stability, near-black frames.
3. **Luminance & Blur** (1 check) — per-frame Tenengrad/luminance classification with segment-level aggregation.
4. **Motion Analysis** (2 checks) — camera stability via single-pass LK optical flow at 0.5x (GPU-accelerated when CUDA OpenCV available). LK is computed inline with frame extraction by a stateful `MotionAnalyzer`; Phase 2 stability + frozen-segment checks slice results from the analyzer without re-decoding the video. Frozen segments derived from LK signal (near-zero translation + rotation).
5. **ML Detection** (6 checks) — face presence (Phase 1 per-frame gate + Phase 2 strict segment check), participants, hand visibility, hand-object interaction, view obstruction, POV-hand angle.

See [checks.md](../checks.md) for full acceptance conditions and thresholds.

---

## 2. The 6 ML Detection Checks

| # | Check | Criterion | Model |
|---|---|---|---|
| 1 | Face Presence (Phase 1 gate) | Per-frame: face detection confidence < 0.8. Bad runs > `--min-bad-segment` s are excluded from checkable segments. | SCRFD-2.5GF |
| 1b | Face Presence (Phase 2 strict) | Zero sampled frames in the segment contain a face at conf ≥ 0.8 (no tolerance for brief flashes that survive Phase 1). | SCRFD-2.5GF (detections reused from Phase 1 cache) |
| 2 | Participants | 0 other persons (face or body parts) in ≥ 90% frames | YOLO11m + SCRFD |
| 3 | Hand Visibility | Both hands fully in frame in ≥ 80% frames **OR** at least one hand fully in frame in ≥ 90% frames (bbox clear of every edge, ≥ 0.7 conf) | Hands23 |
| 4 | Hand-Object Interaction | Interaction detected in ≥ 70% frames | Hands23 (contact state) |
| 5 | View Obstruction | ≤ 10% frames obstructed | OpenCV heuristic (no ML) |
| 6 | POV-Hand Angle | Angle from center to hands < 40° in ≥ 80% frames | Geometric computation on Hands23 output |

---

## 3. Model Selection Rationale

### Constraints (updated from initial brief)
- **Prefer open-source** but do NOT compromise on quality, accuracy, scale, or latency
- **CPU preferred, GPU acceptable** if quality/accuracy/scale/latency demand it
- Must work locally for testing AND on AWS (ECS with GPU, SageMaker)
- No paid per-call APIs (no Google Vision, no Video Intelligence API)
- Models must be testable locally and compatible with AWS managed infrastructure

### Why These Models Were Chosen

#### SCRFD-2.5GF (Face Detection — Check 1)
- **Why over MediaPipe:** 93.8% AP on WIDER Face hard set. MediaPipe is optimized for front-facing webcam use, not egocentric angles where bystander faces appear at unusual angles (profile, top-down, partial occlusion).
- **Why not YOLO:** COCO has no "face" class. SCRFD is purpose-built, tiny (~3MB ONNX), and faster than any general detector for faces.
- **Package:** `insightface` + `onnxruntime`
- **Speed:** ~9.6ms/frame CPU

#### YOLO11m (Person/Object Detection — Checks 2, 5)
- **Why YOLO11m over nano:** 51.5 vs 39.5 mAP on COCO. Person detection is safety-critical (false negatives = missed people in privacy-sensitive data). 22% fewer params than YOLOv8m with higher accuracy.
- **Why not RT-DETR:** Heavier for comparable accuracy, less mature deployment ecosystem.
- **Egocentric note:** Camera wearer is invisible but their arms/hands appear. Pipeline filters out wearer's own body parts by checking overlap with Hands23 hand detections and bottom-center frame position.
- **Package:** `ultralytics`
- **Speed:** ~88ms/frame CPU

#### Hands23 (Hand Detection — Checks 3, 4, 7) [Default]
- **Why Hands23 over 100DOH:** NeurIPS 2023 successor to 100DOH. Trained on 250K images including EPIC-KITCHENS and VISOR egocentric datasets. No custom C++ compilation needed (pure Detectron2). ~3.5x faster than 100DOH on CPU.
- **Why over MediaPipe:** MediaPipe Hand Landmarker has documented AP50 of 29-97% on egocentric data (wildly inconsistent). Hands23 was **trained specifically on egocentric hand images**.
- **Why not HaMeR:** ViT-H backbone (~630M params, ~200-500ms/frame GPU) is overkill for "are hands visible?" detection. HaMeR reconstructs full 3D mesh — we only need bounding boxes + confidence + contact state.
- **Key advantage:** Returns hand contact state (N/S/O/P/F) directly — eliminates need for crude bounding-box overlap heuristic for hand-object interaction detection. Hand bounding boxes are also used for in-frame visibility checks (bbox edge clearance ≥ 2 px).
- **Contact states:** N=no contact, S=self, O=other person, P=portable object, F=stationary object. States P and F = interaction.
- **Architecture:** Faster R-CNN with X-101-FPN backbone, custom heads for contact state, hand side, and grasp type classification.
- **Package:** Custom repo (`github.com/EvaCheng-cty/hands23_detector`) + `detectron2`
- **Speed:** ~1,400ms/frame CPU, **expected ~100-200ms/frame on GPU**

#### 100DOH hand_object_detector (Hand Detection — Legacy)
- **Predecessor to Hands23.** Trained on 131 days of real-world egocentric footage. 90.46% Box AP.
- Requires custom C++ extension compilation (fragile across Python/PyTorch versions).
- **Speed:** ~5,141ms/frame CPU (~3.5x slower than Hands23).
- Available via `./validate.sh --100doh` or `python bachman_cortex/models/download_models.py --100doh`.

#### View Obstruction Heuristic (Check 5)
- **No ML model.** Combines 4 signals on central 80% of frame:
  1. Low spatial variance (std_dev < 15 → homogeneous/covered)
  2. Low edge density (Laplacian variance < 20)
  3. Color channel uniformity (dominant bin > 80% of pixels)
  4. Brightness anomaly (mean < 15)
- Rule: Frame obstructed if ≥ 2 of 4 signals trigger.
- **Speed:** <1ms/frame

---

## 4. Models That Were Considered and Rejected

| Model | Considered For | Reason Rejected |
|---|---|---|
| MediaPipe Face Detector | Check 1 (Face) | Lower accuracy on WIDER Face hard set vs SCRFD. Optimized for webcam, not egocentric angles. |
| MediaPipe Hand Landmarker | Checks 3, 4, 7 | AP50 of 29-97% on egocentric data — wildly inconsistent. Not reliable for 90% threshold on egocentric video. |
| YOLOv8n / YOLO11n | Checks 2, 5 | 39.5 mAP vs 51.5 for medium. Person detection is safety-critical — accuracy matters more than nano's speed advantage. |
| HaMeR (Hand Mesh Recovery) | Check 3 | ViT-H backbone ~630M params, ~200-500ms/frame GPU. Overkill — reconstructs full 3D mesh when we only need bbox + confidence. |
| EgoHOS | Check 4 | Swin-L backbone ~197M params, ~150-300ms/frame GPU. Pixel-level segmentation overkill for binary "is there interaction?" signal. |
| Ego4D HOI models (SlowFast) | Check 4 | Designed for temporal localization ("when does state change?"), not frame-level binary interaction detection. Can't be directly used. |
| RT-DETR | Check 2 | Heavier than YOLO11 for comparable accuracy, less mature deployment. |
| InsightFace SCRFD (larger variants) | Check 1 | SCRFD-2.5GF sufficient. Larger variants (10GF, 34GF) offer marginal accuracy gain at 4-10x cost. |
| FrankMocap | Check 3 | Research code, not production-ready. Slower than 100DOH for similar accuracy. |
| SAM 2 | None | Segmentation model — not needed for any of our detection/classification checks. |
| Custom YOLO face model | Check 1 | Requires training. SCRFD already SOTA for faces at this compute budget. |
| Autoencoder anomaly detection | Check 5 | Requires training on data distribution. Overkill for physical lens blockage detection. |

---

## 5. Ego4D & Egocentric Context

This dataset is directly analogous to **Ego4D** (Meta's 3,670-hour egocentric video benchmark). Key references:

- **Ego4D Hands & Objects benchmark** — defines tasks for hand-object interaction in egocentric video. However, the challenge winners use temporal models (SlowFast, CLIP-based) for "when does state change?" — not frame-level "is the hand touching something right now?" which is what we need.
- **100 Days of Hands (100DOH)** — the 100K+ego dataset (131 days of real-world footage) is the most directly relevant training data for our hand detection needs.
- **EgoHOS (ECCV 2022)** — egocentric hand-object segmentation. Good research but too heavy for our throughput requirements.
- **Epic-Kitchens** — egocentric kitchen activity dataset. Useful for sourcing test videos.

**Key insight:** Egocentric-specific models (100DOH, EgoHOS) significantly outperform general-purpose models (MediaPipe, YOLO pose) on egocentric hand detection because hands are viewed from above/behind, often holding tools, with various skin tones and gloves.

---

## 6. GPU vs CPU Economics

| | GPU (g5.xlarge, A10G) | CPU (c6i.2xlarge) |
|---|---|---|
| Per-video time | ~50s | ~27 min |
| Instances for 5K/day | 3 | 10-17 |
| Daily cost (spot) | ~$29 | ~$139 |
| Monthly cost | ~$870 | ~$4,170 |

**GPU is ~4.8x more cost-effective at scale.** Hands23 is the bottleneck — it's ~1.4s/frame on CPU vs ~100-200ms expected on A10G GPU. (Legacy 100DOH was ~5s/frame CPU.)

**Recommended AWS architecture:** ECS on g5.xlarge (A10G) with Celery workers. Single container with all 4 models in GPU memory (~1.1GB model weights, ~3-4GB total GPU memory). A10G has 24GB — plenty of headroom. Spot instances at ~$0.30-0.50/hr.

---

## 7. Benchmark Results (CPU, macOS M-series ARM64)

Tested on 2026-03-30 with synthetic 30s 1080p test video.

| Model | p50 ms/frame | p95 ms/frame | mean ms/frame | 300 frames est. |
|---|---|---|---|---|
| SCRFD-2.5GF | 9.1 | 11.3 | 9.6 | 2.9s |
| YOLO11m | 86.8 | 101.2 | 88.2 | 26.5s |
| 100DOH (ResNet-101) | 5,095 | 5,567 | 5,141 | 25.7 min |

**Total estimated per 5-min video:**
- CPU: ~27 min (dominated by 100DOH)
- GPU (projected): ~40-65s

---

## 8. Technical Implementation Details

### 100DOH Compilation Fix

The 100DOH hand_object_detector uses custom C++ extensions (ROIAlign, NMS) built against an older PyTorch C++ API. To compile on Python 3.13 + PyTorch 2.11:

**Files patched:**
- `lib/model/csrc/cpu/ROIAlign_cpu.cpp`
- `lib/model/csrc/cpu/nms_cpu.cpp`
- `lib/model/csrc/ROIAlign.h`
- `lib/model/csrc/ROIPool.h`
- `lib/model/csrc/nms.h`

**Changes:**
1. `.type().is_cuda()` → `.is_cuda()`
2. `.type()` → `.scalar_type()` (in `AT_DISPATCH_FLOATING_TYPES`)
3. `.data<T>()` → `.data_ptr<T>()`

After patching, compiled with `python setup.py build develop --no-build-isolation`. Linker warnings about x86_64 architecture (universal binary attempted but torch is arm64-only) are harmless — arm64 version builds and runs correctly.

**Runtime:** Requires `DYLD_LIBRARY_PATH` set to torch lib directory for the C extension to find libtorch at runtime.

### Frame Sampling Strategy

- **1 FPS** (300 frames from 5-min video) — baseline
- Uniform temporal sampling via `cv2.VideoCapture` seeking at 1-second intervals
- Configurable via `PipelineConfig.sampling_fps`
- Frames are resized to `phase1_long_edge` (default 720p) at extraction time
  via `extract_frames(resize_long_edge=...)`. Each decoded frame is resized
  before appending to the output list, so full-resolution frames never
  accumulate in memory. Peak RAM for frames ≈ N × 2.7 MB (720p BGR) instead
  of N × 6.2 MB (1080p BGR).

### Pipeline Architecture

```
Video file
  → Metadata checks (ffprobe, no frames needed)
  → IF metadata fails: return metadata results + SKIPPED for all others
  → Single-pass decode (NVDEC via cv2.cudacodec.VideoReader when available,
    else cv2.VideoCapture) @ 1 FPS sampling. Each decoded frame is resized
    to 720p long-edge (PipelineConfig.phase1_long_edge) before being
    appended to the output list — full-resolution frames never accumulate
    in memory. The resized frame is also fed to a stateful MotionAnalyzer
    (LK optical flow accumulator) at its frame_skip cadence
    (native_fps / target_fps, default target 30).
  → Phase 1 ML batch loop:
      1. YOLO11s batched object detection (16-frame batches)
      2. SCRFD face detection dispatched across batch via ThreadPool
         (PipelineConfig.scrfd_threads, default 4)
      3. Hands23 hand-object detection (per frame, Detectron2 predictor)
  → Phase 1 evaluation: face + participant per-frame pass/fail →
    bad segments → overlap merge → good segments → filter by min duration
  → Checkable segments → Phase 2 per-segment validation:
      - Luminance & blur (Tenengrad + luminance decision table)
      - View obstruction heuristic
      - Hand visibility / hand-object interaction / POV-hand angle
        (reused from Phase 1 cache)
      - Face presence (strict, zero-tolerance — reused SCRFD cache)
      - Motion stability + frozen segments (sliced from MotionAnalyzer,
        no re-decode)
  → Phase 3 yield calculation
```

### MotionAnalyzer (single-pass LK, 2026-04-13)

`bachman_cortex/checks/motion_analysis.py` defines a `MotionAnalyzer`
dataclass that accumulates per-second LK translation / rotation arrays
plus per-sampled-frame frozen state as `frame_extractor.extract_frames`
walks the video. The analyzer is created once in
`ValidationProcessingPipeline.process_video` before extraction and passed
to the extractor via the `motion_analyzer` kwarg. Extraction resizes each
decoded frame to 720p first, then calls
`analyzer.process_frame(frame_bgr, frame_idx)` at the analyzer's
`frame_skip` cadence — the analyzer applies its own `fast_scale` (0.5×)
on top, so LK runs at ~360p.

Phase 2's motion check is `check_motion_combined_from_analyzer(analyzer,
start_sec, end_sec, ...)` — it derives stability + frozen results from
the pre-populated analyzer, no `cv2.VideoCapture` re-open. LK numerical
path is identical to the original `check_motion_combined` (same
`_lk_track`, same `_transforms_to_score` scoring).

Replaces the prior design where Phase 1 did a full decode and Phase 2
re-decoded each segment's range on CPU. Observed impact on a 501 s /
2336×1080 LATAM sample: total 133 s → 116 s (default config, after
CUDA OpenCV and NVDEC land).

### Custom CUDA OpenCV + NVDEC (2026-04-13)

PyPI's `opencv-python` ships without CUDA. To unlock
`cv2.cuda.SparsePyrLKOpticalFlow` (used by the MotionAnalyzer) and
`cv2.cudacodec.VideoReader` (NVDEC hardware decode used by
`frame_extractor.extract_frames`), the project ships a build script:

`scripts/install_opencv_cuda.sh`

The script:
- `apt install`s CUDA toolkit + build deps (needs sudo password once).
- Expects `NVCODEC_SDK` env var pointing at an extracted NVIDIA Video
  Codec SDK 12.x tree (user downloads from developer.nvidia.com — behind
  a free login). Copies `nvcuvid.h`, `cuviddec.h`, `nvEncodeAPI.h` to
  `/usr/include/` and the stub libs to `/usr/lib/x86_64-linux-gnu/`.
- Clones OpenCV 4.10 + opencv_contrib, configures with `WITH_CUDA=ON`,
  `WITH_CUDNN=OFF`, `WITH_NVCUVID=ON`, `WITH_NVCUVENC=ON`.
- Pins CUDA host compiler to `gcc-12` (CUDA 12.0's nvcc caps at gcc 12;
  Ubuntu 24.04's default `cc` is gcc 13, which nvcc rejects).
- Sets `CMAKE_POLICY_VERSION_MINIMUM=3.5` (CMake 4.x dropped support for
  OpenCV 4.10's `cmake_minimum_required(VERSION 2.8)` in some sub-scripts).
- Disables `BUILD_opencv_dnn` and `BUILD_opencv_mcc` — the OpenCV 4.10
  pyopencv bindings for dnn (pyopencv_dnn.hpp referencing `dnn::`
  without namespace qualifier) and mcc (abstract `cv::mcc::CChecker`
  passed to `makePtr`) fail to compile on this host toolchain. Both
  modules are unused by the project (we use ORT / PyTorch for inference
  and don't need color calibration).
- Disables `OPENCV_GENERATE_PKGCONFIG` (CMake 4.x compat).
- Pre-resizes rather than ninja-installing `opencv4.pc`.

After the build the venv's cv2 module has:
- `cv2.cuda.SparsePyrLKOpticalFlow` — picked up automatically by
  `motion_analysis._CUDA_LK` at import time.
- `cv2.cudacodec.VideoReader` — detected at import time by
  `frame_extractor._nvdec_available()`. When true, `extract_frames`
  opens the video via `cv2.cudacodec.createVideoReader`, iterates
  `nextFrame()`, converts BGRA → BGR on GPU (`cv2.cuda.cvtColor`),
  downloads to CPU only at sample / motion points.
- `extract_frames` metadata dict includes `backend: "nvdec" | "cpu"` to
  tell consumers which path ran.

### `cv2.dnn` compat shim (2026-04-13)

Because the custom OpenCV build disables `cv2.dnn`, the only project
dependency that relies on `cv2.dnn.blobFromImage` — `insightface`'s SCRFD
preprocessing — would crash on the new cv2.

`bachman_cortex/_cv2_dnn_shim.py` provides pure-numpy implementations of
`blobFromImage` and `blobFromImages` (resize → optional BGR→RGB swap →
mean-subtract → scale → NCHW float32 layout). `bachman_cortex/__init__.py`
imports the shim and calls `install()`, which attaches a `cv2.dnn`
module object with those two functions *only if* `cv2.dnn` is not
already present (i.e. no-op on the standard pip wheel).

The shim's signature mirrors OpenCV's C++ `blobFromImage`. Because
insightface never calls DNN inference via cv2 (it uses ONNX Runtime),
replacing only the preprocessing helpers is sufficient.

### Check 2 Participants — Wearer Filtering

The camera wearer's arms/hands appear in frame but should NOT be counted as "another person." Filtering logic:
1. Exclude YOLO person detections that overlap significantly with Hands23 hand detections (wearer's hands = wearer's arms)
2. Exclude detections anchored at bottom-center of frame (wearer's torso/arms)
3. Require minimum person bbox height > 15% of frame height (filter tiny partial detections)
4. Combine with SCRFD face detections — even a disembodied face with no body detection counts as "another person present"

---

## 9. Project Structure

```
hl-bachman/
├── validate.sh                         # One-command entry point
├── pyproject.toml                      # Package configuration (hl-video-validation)
├── checks.md                           # Check specifications and thresholds
├── README.md
├── scripts/
│   └── install_opencv_cuda.sh          # Build custom OpenCV with CUDA + NVDEC
└── bachman_cortex/
    ├── __init__.py                     # Installs cv2.dnn shim before any import
    ├── _cv2_dnn_shim.py                # Pure-numpy blobFromImage / blobFromImages
    ├── pipeline.py                     # ValidationProcessingPipeline orchestrator
    ├── run_batch.py                    # CLI entry point (hl-validate)
    ├── data_types.py                   # Shared dataclasses
    ├── reporting.py                    # Per-video + batch report generation
    ├── CONTEXT.md                      # This file
    ├── checks/
    │   ├── check_results.py            # CheckResult dataclass
    │   ├── video_metadata.py           # 6 metadata checks
    │   ├── luminance_blur.py           # Tenengrad + luminance decision table
    │   ├── motion_analysis.py          # MotionAnalyzer (single-pass LK) + slicer
    │   ├── participants.py             # Person count check
    │   ├── hand_visibility.py          # Hand detection check
    │   ├── hand_object_interaction.py  # Contact state check
    │   ├── view_obstruction.py         # Lens obstruction heuristic
    │   ├── pov_hand_angle.py           # Hand angle check
    │   └── face_presence.py            # Strict segment-level face presence
    ├── models/
    │   ├── download_models.py          # Model weight downloader
    │   ├── scrfd_detector.py           # SCRFD (cuDNN HEURISTIC autotune)
    │   ├── yolo_detector.py            # YOLO11s object detector
    │   ├── hand_detector.py            # Hands23 hand-object detector
    │   └── weights/                    # ~440MB, downloaded on first run
    └── utils/
        ├── frame_extractor.py          # NVDEC-or-VideoCapture decode + motion tee
        ├── video_metadata.py           # FFprobe metadata extraction
        ├── segment_ops.py               # Segment merging / filtering
        └── early_stop.py                # Per-frame eval helpers
```

**YOLO weights** (`yolo11s.pt`) are downloaded by ultralytics on first run (~20MB).

---

## 10. Python Dependencies

```
torch>=2.0
torchvision>=0.15
onnxruntime>=1.17.0
insightface>=0.7.3
ultralytics>=8.1.0
transformers>=4.36.0
supervision>=0.18.0
opencv-python>=4.9.0
numpy>=1.24.0
Pillow>=10.0
tqdm>=4.65
```

**Special:** Hands23 requires `detectron2` (installed from GitHub with `--no-build-isolation`). Legacy 100DOH additionally requires C++ extensions built from source. System dependency: `ffmpeg`/`ffprobe` for video metadata extraction. All handled automatically by `validate.sh`.

---

## 11. Next Steps

1. **Test with real egocentric videos** — synthetic data validates the pipeline but real data is needed to:
   - Tune confidence thresholds for each check
   - Measure false positive/negative rates
   - Validate 100DOH hand detection accuracy on agricultural/commercial/residential tasks

2. **GPU testing** — 100DOH at 5s/frame on CPU is the bottleneck. Need to test on A10G GPU to confirm ~100-150ms/frame.

3. **Threshold tuning** — all thresholds are set to idea-brief spec values. May need adjustment based on real data:
   - Face confidence threshold (0.8)
   - Person confidence threshold (0.4)
   - Hand confidence threshold (0.7)
   - Obstruction heuristic thresholds

4. **Containerization:** Create Docker image with all models + CUDA for AWS ECS deployment.

5. **Integration:** Wire pipeline into Celery workers for the FastAPI backend described in the idea brief.

---

## 12. Key Decisions & Rationale Log

| Decision | Rationale |
|---|---|
| GPU over CPU at scale | 4.8x more cost-effective ($870 vs $4,170/month) |
| Hands23 over 100DOH | NeurIPS 2023 successor, 250K training images, no C++ compilation, ~3.5x faster on CPU |
| Hands23 over MediaPipe for hands | MediaPipe has 29-97% AP on egocentric data; Hands23 trained on egocentric datasets |
| SCRFD over MediaPipe for faces | 93.8% AP on WIDER hard set; MediaPipe optimized for webcam |
| Hands23 input downscaling (720p default) | Reduces inference time on the most expensive model with negligible accuracy impact |
| Heuristic over ML for view obstruction | No standard ML model exists; signal is fundamentally low-level |
| YOLO11m over YOLO11n | 51.5 vs 39.5 mAP; person detection is safety-critical for privacy |
| Metadata gate (short-circuit) | Avoids expensive ML inference on videos that fail basic format/duration requirements |
| Farneback over Lucas-Kanade for stability | Dense optical flow gives more accurate camera motion estimate |
| Native FPS for frozen segments | 30 consecutive frames = 1 second; sampled frames would miss short freezes |
| 1 FPS sampling for ML checks | 120 frames gives sufficient temporal coverage; configurable for GPU |
| Single-pass decode (2026-04-13) | Phase 1 extraction and Phase 2 motion analysis share one cv2 read pass. MotionAnalyzer accumulates per-second LK trans/rot inline; Phase 2 slices without re-decoding. Cuts LATAM 501s video 133s → 116s default config. |
| Phase 1 720p downscale (2026-04-13, moved to extraction 2026-04-16) | Frames resized to 720p long-edge at decode time inside `extract_frames(resize_long_edge=...)`. Previously a separate pass in `_phase1_run_inference` that created a second list while the 1080p originals were still alive — caused OOM on videos longer than ~30 min (dual 1080p + 720p lists exceeded 16 GB RAM). Now only 720p frames exist in memory at any point. MotionAnalyzer also receives 720p input (applies its own 0.5× downscale to 360p for LK). |
| Threaded SCRFD (2026-04-13) | `FaceAnalysis.get()` isn't batch-capable and releases the GIL inside ORT `run()`. 4-worker ThreadPoolExecutor overlaps SCRFD with YOLO / Hands23 in the inner loop (~1.2× Phase 1 win). |
| cuDNN HEURISTIC on SCRFD (2026-04-13) | ORT defaults to `cudnn_conv_algo_search=EXHAUSTIVE` which adds 2-3s to first inference per new input shape. HEURISTIC smoothes p99 at negligible throughput cost. |
| Custom OpenCV-CUDA build (2026-04-13) | PyPI cv2 has no CUDA. Building from source unlocks `cv2.cuda.SparsePyrLKOpticalFlow` (GPU motion) and `cv2.cudacodec.VideoReader` (NVDEC decode). Drops LATAM 501s video extraction 46.6s → 32.8s. Cost: ~60 min one-time build via scripts/install_opencv_cuda.sh. |
| Skip cv2.dnn / cv2.mcc in custom build (2026-04-13) | OpenCV 4.10's pyopencv_dnn.hpp and opencv_contrib/mcc bindings fail to compile on CMake 4.x + gcc-12. Neither module is used by the project; insightface's blobFromImage is reproduced in `_cv2_dnn_shim.py`. |
| Removed privacy, body-part-visibility, and egocentric checks (2026-04-15) | Three checks retired and their code/models deleted. Grounding DINO (privacy), YOLO11m-pose (body-part-visibility), and the egocentric-perspective heuristic are all gone. Face presence was previously the opt-in replacement for egocentric; it is now an always-on Phase 2 strict check (zero sampled frames with face conf ≥ 0.80) that reuses SCRFD detections from Phase 1. Simplifies config (no opt-in flags remain for ML checks), drops ~1.1 GB of model weights, and shrinks the surface area of Phase 1 / Phase 2 to the checks that actually run. |
| Timestamps-only reporting (2026-04-14) | Dropped ffmpeg clip extraction entirely (`clip_extractor.py` removed). Per-video `report.md` is now a unified chronological `HH:MM:SS.mmm` timeline labeling each contiguous range **USABLE / UNUSABLE / REJECTED** with failing-check reasons and observed metrics. Metadata-fail videos render as one REJECTED row spanning the whole video. Dataclasses renamed (`PreFilteredClip` → `CheckableSegment`, `ClipValidationResult` → `SegmentValidationResult`). `--min-segment` default raised 30→60s so Phase 2 pass-rate metrics stay statistically meaningful; clean gaps below the threshold render as `UNUSABLE (segment_too_short)`. |

---

*End of context document*
