# Codebase Concerns

**Analysis Date:** 2026-04-15

## Tech Debt

### Grounding DINO is an architectural bottleneck (Medium Priority)

**Issue:** Grounding DINO (200M-parameter foundation model) spends 112.5s (50% of pipeline) detecting 4 fixed object classes in the privacy check, while YOLO11s detects 80 COCO classes in 1.2s. The model's zero-shot strength is completely wasted on fixed-class detection.

**Files:** `bachman_cortex/models/grounding_dino_detector.py`, `bachman_cortex/pipeline.py:407-413` (GDINO thread synchronization)

**Impact:** Privacy check is disabled by default (`--privacy-check` opt-in) to avoid regressing default-config throughput. When enabled on 180s clips, doubles processing time (~100s → ~200s).

**Fix approach:** Replace GDINO with a fine-tuned YOLO11n on 6 privacy classes (credit_card, id_card, pii_document, screen_mobile, screen_laptop, screen_monitor). Bootstrap training data from GDINO detections on real egocentric footage (~500-1000 examples/class). Expected result: ~112.5s → ~2-3s. Detailed plan in `YOLOV5N_FINETUNE_PLAN.md` (phases 1-5 documented; phase 6 integration template ready).

---

### Motion analysis still re-opens video per clip (Latent, now fixed)

**Issue:** Motion analysis in Phase 2 used to re-open the video file via `cv2.VideoCapture` for each clip, causing ~2x CPU decode overhead (one full decode in Phase 1 extraction, another in Phase 2).

**Files:** `bachman_cortex/checks/motion_analysis.py:597` (old code)

**Status:** FIXED (2026-04-13). Replaced with `MotionAnalyzer` stateful accumulator that tees the native-rate decode stream inline during extraction. Phase 2 motion checks slice pre-computed results without re-decoding.

**Observed impact:** LATAM 501s clip: extraction 46.6s → 32.8s (with NVDEC), end-to-end 133s → 116s (13% total cut).

---

### Phase 1 frame cache still holds full-resolution frames (Fixed, resizing pending validation)

**Issue:** Phase 1 used to cache all native-resolution frames (e.g., 1.1GB for 180 frames @ 1080p) in memory until Phase 2 completed.

**Files:** `bachman_cortex/pipeline.py:654-659` (cache structure)

**Status:** FIXED (2026-04-13). Phase 1 downscales all frames to 720p long-edge at entry (`PipelineConfig.phase1_long_edge=720` default); cache stores 720p frames only. Phase 1 peak memory halved (~1.1GB → ~475MB).

**Risk:** Face recall on LATAM samples must be re-validated at 720p. No regressions observed in smoke tests so far. Fallback: set `phase1_long_edge=None` to disable, or `phase1_long_edge=900` for middle-ground.

---

### GDINO parallel thread synchronization is brittle (Medium Priority)

**Issue:** GDINO runs in a background thread gated on `_gdino_skip_ready` event (`pipeline.py:407-413`). The event is set **after** the face/participant inner loop completes, forcing GDINO to wait until Phase 1 is done, then run sequentially. Even when privacy is enabled, GDINO is not truly parallel.

**Files:** `bachman_cortex/pipeline.py:407-413, 555-557`

**Impact:** GDINO processing adds wall-clock time sequentially (when enabled). Intended as an optimization but doesn't achieve the goal.

**Fix approach:** Move `_gdino_skip_ready.set()` to just after YOLO batch completes (not after the full face/participant loop). Alternatively, populate the skip set lazily as face/participant results become available, allowing GDINO to start earlier.

---

### Phase 1 inner loop still serializes SCRFD and Hands23 (Partially optimized)

**Issue:** YOLO gets batched inference; SCRFD and Hands23 remain per-frame on a single CUDA stream. The Phase 1 inner loop processes 16-frame batches serially for face/hand detection.

**Files:** `bachman_cortex/pipeline.py:497-506` (inner loop), `bachman_cortex/pipeline.py:499-500` (SCRFD dispatch)

**Status:** PARTIALLY OPTIMIZED (2026-04-13). SCRFD now dispatches to a `ThreadPoolExecutor` (4 workers default), but Hands23 remains per-frame. Expected ~1.2x Phase 1 speedup from SCRFD overlap.

**Remaining work:** Batch Hands23 detection per frame group if the Detectron2 predictor API supports it. Currently unprofiled; may already be near-optimal given Hands23's Faster R-CNN architecture.

---

### SSL certificate verification disabled for model downloads (Security Low Risk)

**Issue:** Hands23 weights download disables SSL verification (`--no-check-certificate` in wget, `ctx.check_hostname = False` + `ctx.verify_mode = ssl.CERT_NONE` in urllib).

**Files:** `bachman_cortex/models/download_models.py:74-84`

**Why:** The fouheylab.eecs.umich.edu SSL certificate is not recognized by some Python environments (certificate chain issue).

**Mitigation:** This is a one-time download, not a production attack surface. The project runs locally only (no exposed APIs). Download happens at CLI invocation or in CI — not in production inference. Hash-verify the downloaded weights file size (`weight_file.stat().st_size / 1024 / 1024`) matches the expected 400MB before continuing.

**Better fix:** Contact fouheylab to fix their certificate, or mirror weights to a CDN with valid chain. For now, add a warning + size-check assertion.

---

### OpenCV-CUDA build is fragile (Medium Risk)

**Issue:** The custom OpenCV build requires:
  1. CUDA toolkit + build deps (must be installed with sudo)
  2. NVIDIA Video Codec SDK 12.x extracted to a known path (user-provisioned, not automated)
  3. CMake 4.x compatibility patches (CMAKE_POLICY_VERSION_MINIMUM=3.5)
  4. `gcc-12` pinning (CUDA 12.0 nvcc rejects gcc 13+)
  5. `BUILD_opencv_dnn=OFF` due to unresolved `dnn::` namespace (OpenCV 4.10 + CMake 4.x bug)

**Files:** `scripts/install_opencv_cuda.sh`, `bachman_cortex/_cv2_dnn_shim.py` (shim workaround)

**Impact:** One-time ~60 minute build per host. Falls back gracefully to CPU LK + CPU ffmpeg decode if CUDA path is unavailable.

**Fragility:**
  - If CUDA toolkit or Codec SDK is not installed, the build silently skips CUDA features rather than failing loud.
  - The shim (`_cv2_dnn_shim.py`) patches `cv2.dnn` if DNN module is missing — correct, but unusual pattern that could confuse future maintainers.
  - CMake 4.x support in OpenCV 4.10 is borderline; later OpenCV 4.11+ may break the script.

**Mitigation:** Document the one-time cost in README (done). Add CI/docker image pre-built with CUDA. Ship binary wheels for common distributions.

---

### `cv2.dnn` shim is an unusual compatibility pattern (Low Risk, Documented)

**Issue:** When the custom OpenCV build disables `BUILD_opencv_dnn`, the only project dependency that needs `cv2.dnn.blobFromImage` — insightface's SCRFD preprocessing — would fail. The project works around this by installing a pure-numpy shim at module import time.

**Files:** `bachman_cortex/_cv2_dnn_shim.py`, `bachman_cortex/__init__.py` (calls `install()`)

**Pattern:** Monkey-patches `cv2.dnn` module if functions are missing. Idempotent and backward-compatible.

**Risk:** This is a fragile compatibility pattern that assumes:
  1. insightface calls `cv2.dnn.blobFromImage` (not `cv2.dnn_blobFromImage` or some variant)
  2. insightface never calls DNN inference via cv2 (it uses ORT instead)
  3. Future versions of insightface don't add new cv2.dnn dependencies

**Mitigation:** Already documented in `CONTEXT.md:314-329`. Add runtime assertion at pipeline start to verify `cv2.dnn.blobFromImage` exists and is callable.

---

## Known Bugs

### CUDA LK optical flow initialization fails on some hosts (Latent, Auto-Fallback)

**Symptoms:** `motion_analysis._CUDA_LK` = False even when `cv2.cuda.getCudaEnabledDeviceCount() > 0`. Motion analysis runs on CPU instead of GPU.

**Files:** `bachman_cortex/checks/motion_analysis.py:62-68`

**Trigger:** The smoke test `cv2.cuda.SparsePyrLKOpticalFlow.create()` raises `cv2.error` (caught silently), causing `_CUDA_LK` to remain False.

**Workaround:** Pipeline automatically uses CPU LK as fallback. User can check whether GPU acceleration is active by examining logged extraction metadata (`backend: "nvdec" | "cpu"`).

**Investigation needed:** Print the specific `cv2.error` message to help diagnose why CUDA initialization fails. Current code swallows the exception.

---

### NVDEC frame download is GPU↔CPU sync point (Performance, Known)

**Issue:** Even with NVDEC hardware decode, frames must be downloaded from GPU to CPU at sampling points (`frame_extractor.py:113`). This is a GPU↔CPU synchronization point that can stall the pipeline if CUDA operations are not properly pipelined.

**Files:** `bachman_cortex/utils/frame_extractor.py:113` (download call)

**Impact:** Measured benefit of NVDEC is ~13% total (extraction 46.6s → 32.8s on LATAM 501s clip), not the 3-4x that pure NVDEC decoding might suggest. CPU is likely blocking on GPU downloads.

**Investigation:** Profile with `nvprof` or `nsys` to check for stalls. Possible fix: pre-allocate pinned CPU memory buffers and use CUDA async downloads.

---

### FFprobe subprocess call has no timeout (Low Risk)

**Issue:** `video_metadata.py:23` calls `subprocess.run(ffprobe_cmd)` with no timeout. If ffprobe hangs (corrupted video, slow filesystem), the pipeline blocks indefinitely.

**Files:** `bachman_cortex/utils/video_metadata.py:23`

**Impact:** Batch processing could hang per video.

**Fix:** Add `timeout=30` parameter to subprocess.run.

---

## Security Considerations

### Model download URLs are unvalidated (Low Risk, HTTP only)

**Issue:** Model weights are downloaded from URLs without hash verification:
  - `insightface`: Fetched dynamically by `FaceAnalysis` (vendor-controlled)
  - YOLO11s: Fetched dynamically by Ultralytics (vendor-controlled)
  - Hands23: Manual URL in `download_models.py:72` (HTTP, fouheylab.eecs.umich.edu)
  - 100DOH: Google Drive URL via gdown (redirects)

**Files:** `bachman_cortex/models/download_models.py:72,176`

**Mitigation:** All downloads are one-time, run locally, by trusted developers. No production inference downloads models. Cache model files after first download — weights directory is never re-downloaded unless manually deleted.

**Better fix:** Publish SHA256 hashes of expected weights (20 bytes each) in the repo. Verify after download: `openssl dgst -sha256 weights.pt`.

---

### Path handling in transcode outputs (Medium Risk)

**Issue:** `transcode.py:77` builds output path without validating `video_path` input:
  ```python
  output_path = preprocessed_dir / f"{Path(video_path).stem}.mp4"
  ```

If `video_path` contains a string like `../../../etc/passwd`, the `.stem` extraction is safe, but if a symlink traversal or path traversal attack exists elsewhere, this could be exploited.

**Files:** `bachman_cortex/utils/transcode.py:77`

**Mitigation:** The function already receives `video_path` from trusted CLI sources (collect_videos validates paths). No user-controlled input is interpolated. Risk is low but path should be resolved to absolute path: `Path(video_path).resolve()` before use.

---

### Subprocess calls use shell=False (Secure)

**Issue:** None. All subprocess calls in the codebase use `shell=False` and pass command lists, not strings. This prevents shell injection.

**Files:** `transcode.py:106`, `video_metadata.py:23`, `download_models.py:58,96,156,184`

---

## Performance Bottlenecks

### Hands23 is the residual bottleneck on CPU (Medium Priority)

**Problem:** Hands23 runs at ~1,400ms/frame on CPU. For a 180-frame 1-FPS sample, this is ~210 seconds of the pipeline. Phase 1 + Phase 2 are heavily dominated by Hands23.

**Files:** `bachman_cortex/models/hand_detector.py`, `bachman_cortex/pipeline.py:753-757` (Phase 2 call)

**Cause:** Faster R-CNN backbone + custom Detectron2 architecture. No batching support (per-frame only).

**Improvement path:**
  1. **Short-term (P1 in OPTIMIZATION_PLAN_V3):** Profile Detectron2 predictor's batching capability. May already support batches internally; check if Model Zoo exports support batch inference.
  2. **Medium-term (P2.2):** Swap Hands23 backbone from X-101-FPN (~150M params) to a lighter variant (ResNet-50 or MobileNet). Literature suggests Hands23 accuracy is dominated by the head (contact state classification), not backbone depth. Prototype on the training codebase.
  3. **GPU tier (expected ~100-200ms/frame):** Hands23 is designed to run on modern GPUs. Benchmark on A10G; if it hits <500ms, that's a 7-10x speedup and pivots the bottleneck away from hands.

---

### Metadata gate (`ffprobe`) is always run even for rejected videos (Low Priority)

**Problem:** `pipeline.py:219` always calls `get_video_metadata` even if the video is later rejected at Phase 0. This is unavoidable (metadata gate must happen first).

**Impact:** Negligible (ffprobe is fast, ~50-100ms).

**Note:** This is correct behavior — Phase 0 is intentionally a gate.

---

## Fragile Areas

### Motion analysis is sensitive to frame skip cadence (Medium Priority, Mitigated)

**Component:** `MotionAnalyzer` accumulates per-second LK translation/rotation arrays. The cadence at which frames are processed affects temporal resolution.

**Files:** `bachman_cortex/checks/motion_analysis.py:200-250` (MotionAnalyzer class), `bachman_cortex/pipeline.py:252-260` (initialization)

**Fragility:** If the native video FPS is unknown or incorrect, the per-second granularity breaks. Example: if the video is labeled as 30 FPS but actually plays at 29.97 FPS, the LK samples will accumulate at the wrong cadence.

**Mitigation:** Obtained from `cv2.VideoCapture.get(CAP_PROP_FPS)` which is reliable for standard codecs. For corrupted headers, falls back to metadata from `ffprobe` (more robust). Validated in smoke tests.

**Safe modification:** The `MotionAnalyzer` dataclass is deterministic given frame_skip. If test regression occurs, compare LK samples (sample counts, translation/rotation distributions) between old/new code on the same test video.

**Test coverage:** `bachman_cortex/tests/benchmark_phase_correlation.py` has a motion analyzer bench. Add a regression test that compares motion scores on a fixed test clip.

---

### Segment merging / filtering logic is complex (Medium Priority)

**Component:** `segment_ops.py` merges overlapping bad segments, filters by duration, computes good segments. Contains multiple thresholds.

**Files:** `bachman_cortex/utils/segment_ops.py:43-125` (merge_bad_segments, filter_short_bad_segments, compute_good_segments)

**Fragility:** The logic is correct but non-obvious:
  1. `per_frame_to_bad_segments`: Converts per-frame pass/fail to contiguous bad segments.
  2. `merge_bad_segments`: Merges segments within `gap_threshold` (default 1.0 s).
  3. `filter_short_bad_segments`: Drops segments shorter than `min_duration` (default 2.0 s).
  4. `compute_good_segments`: Inverse of bad segments.
  5. `filter_checkable_segments`: Keeps only segments >= `min_checkable_segment_sec` (default 60 s).

Safe modification: Each function has isolated responsibility. Add unit tests for boundary cases (e.g., bad segment at video start/end, gap exactly at threshold, contiguous bad segments).

**Test coverage:** Gaps. Add `tests/test_segment_ops.py` with synthetic per-frame results.

---

### Hands23 wearer filtering heuristic is context-dependent (Medium Priority, Mitigated)

**Component:** Participants check filters out wearer's hands/arms by checking overlap with Hands23 detections and bottom-frame anchoring.

**Files:** `bachman_cortex/checks/participants.py:80-120` (wearer filtering)

**Fragility:** The heuristic assumes:
  1. Wearer's hands always overlap with Hands23 detections (true for "visible hands" check, but wearer may be out-of-frame)
  2. Other people's detections are not bottom-frame-anchored (assumption breaks for very tall people or overhead angles)
  3. YOLO person bbox height > 15% of frame is a meaningful filter (may fail on close-up torso-only shots)

Safe modification: Use historical data (test set) to validate wearer-filtering accuracy. Flag edge cases in logs for manual review. Consider a `--strict-participants` mode that requires explicit "is this the wearer?" logic (e.g., hand bounding box centroid in bottom 30% of frame).

**Test coverage:** Add test videos with edge cases (wearer's hand only in frame, tall bystander, overhead angle). Verify participants check output.

---

### Grounding DINO OOM fallback is undocumented (Low Risk)

**Component:** GDINO batching has auto-fallback: halves batch size on CUDA OOM, retries down to single-frame.

**Files:** `bachman_cortex/models/grounding_dino_detector.py:140-150` (batch processing with OOM fallback)

**Fragility:** If a single-frame forward pass OOMs, the entire privacy check fails silently (graceful degradation: frames treated as privacy-clean). This is not user-visible.

**Mitigation:** Already documented in CONTEXT.md. Add a warning log if fallback occurs: "GDINO OOM, halving batch size" so users know GPU memory is tight.

---

## Scaling Limits

### Host has high swap thrashing under heavy multi-worker load (High Priority)

**Current state:** (from OPTIMIZATION_PLAN_V3:75)
  - Host: 16 GB RAM, ~9.4 GB free
  - Swap: 4 GB, 92% used (3.7 GB / 4 GB)

**Problem:** Multi-worker batch runs (`--workers 4` or higher) cause each worker to load all models (~6-8 GB) into its own process. Total memory demand = 4 workers × 8 GB = 32 GB, but only 16 GB + 4 GB swap = 20 GB available. This triggers swap thrashing.

**Files:** `bachman_cortex/run_batch.py:70-81` (_auto_detect_workers), `bachman_cortex/pipeline.py:141-167` (load_models)

**Impact:** Processing slows drastically under load. Observed in LATAM batch runs.

**Scaling path:**
  1. **Immediate:** Set `--workers 1` or `--workers 2` by default (even with `_auto_detect_workers` logic).
  2. **Short-term:** Use shared model loading via a model server (Redis cache or a central process that all workers inherit).
  3. **Long-term:** AWS deployment on A10G instances with 24 GB VRAM. Tune workers for that environment separately.

**Safe modification:** Add a memory check at startup: if available memory < (workers × 8 GB + 2 GB buffer), warn the user and suggest `--workers N` override.

---

### LATAM sample set is 2336×1080 (non-standard resolution) (Low Risk)

**Problem:** LATAM test samples have atypical 2336×1080 resolution (ultra-wide aspect ratio). Frame caches and model inputs are sized assuming 1920×1080 or 1280×720 in benchmarks.

**Files:** Phase 1 frame downscaling (`pipeline.py:120-122`) handles arbitrary resolutions, but memory estimates in plans assume 1080p baseline.

**Impact:** Memory usage on ultra-wide videos may be higher than projected. Unlikely to break anything, but should be noted.

---

## Dependencies at Risk

### insightface SCRFD model may be delisted (Low Risk)

**Risk:** InsightFace's buffalo_sc model (SCRFD-2.5GF) is downloaded dynamically by `FaceAnalysis(name="buffalo_sc", root=cache_dir)`. If InsightFace stops hosting this model or changes the URL structure, downloads fail.

**Files:** `bachman_cortex/models/scrfd_detector.py:18-32`, `bachman_cortex/models/download_models.py:18-32`

**Mitigation:** InsightFace is a mature project (maintained by Insightface contributors). Model URLs are stable. As backup, the project could mirror the model weights locally or to S3.

**Better fix:** Add a `--local-models` flag that uses pre-downloaded weights from a models/ directory instead of downloading dynamically.

---

### Ultralytics YOLO models are auto-downloaded at first run (Low Risk)

**Risk:** YOLO11s and YOLO11m-pose are downloaded on first model inference by ultralytics (from `download.ultralytics.com`). If Ultralytics changes their CDN or domain, downloads fail.

**Files:** `bachman_cortex/models/yolo_detector.py:13-25`, `bachman_cortex/models/yolo_pose_detector.py:13-30`

**Mitigation:** Ultralytics is the official maintainer of YOLO models; URLs are stable. The project could pre-download weights and vendor them.

---

### Detectron2 build-from-source dependency (Medium Risk)

**Risk:** Hands23 requires Detectron2, which is installed via `pip install git+https://github.com/facebookresearch/detectron2.git --no-build-isolation`. This clones from GitHub and builds C++ extensions locally.

**Files:** `bachman_cortex/models/download_models.py:96-100`

**Impact:** If GitHub is inaccessible or the repo is deleted, Hands23 setup fails. Build times are long (~2-3 minutes). C++ compilation failures are hard to debug.

**Mitigation:** This is documented as a one-time setup cost. For production, pre-build a Docker image with Detectron2 already compiled.

**Better fix:** Use a precompiled Detectron2 wheel from a Python package index (if available) rather than building from source.

---

## Missing Critical Features

### No hash verification of downloaded model weights (Low Risk)

**Problem:** Model weights are downloaded without verifying file integrity. A corrupted download or MITM attack could load malicious model files.

**Files:** All `download_*` functions in `bachman_cortex/models/download_models.py`

**Blocks:** Nothing critical — models are verified by loaded models being used (if weights are corrupted, model inference fails with a clear error). But security best practice would add hash checks.

**Improvement:** Publish SHA256 hashes in a `models/CHECKSUMS.txt` file. After download, verify `sha256sum weights.pt` matches the expected hash.

---

### No explicit test coverage for segment ops (Medium Priority)

**Problem:** The segment merging, filtering, and good-segment computation logic in `segment_ops.py` is complex and has no unit tests.

**Files:** `bachman_cortex/utils/segment_ops.py`

**Blocks:** Nothing critical — logic is covered implicitly by end-to-end tests (via `run_batch.py` on test videos). But regressions in edge cases (video start/end boundary segments, exact-threshold gap merges) could go unnoticed.

**Test coverage needed:** Add `bachman_cortex/tests/test_segment_ops.py` with synthetic per-frame results and expected segment outputs.

---

### No smoke test for Hands23 wearer filtering (Low Priority)

**Problem:** The wearer filtering heuristic in `participants.py` is never validated against known ground truth (videos where wearer is/is not visible, etc.).

**Files:** `bachman_cortex/checks/participants.py:80-120`

**Blocks:** Nothing — heuristic is empirically sound (tested on real LATAM data). But a formal validation set would catch edge cases.

**Test coverage needed:** Curate 10 LATAM clips covering edge cases (wearer visible, wearer mostly out-of-frame, wearer hands-only, wearer with tall bystander). Validate participants check output.

---

## Test Coverage Gaps

### Hands23 accuracy on LATAM data is unmeasured (High Priority)

**What's not tested:** Hands23's per-frame detection accuracy (precision/recall) on the specific LATAM video set (agricultural/commercial/residential tasks). The model was trained on egocentric kitchen / YouTube videos; generalization to other domains is unknown.

**Files:** `bachman_cortex/models/hand_detector.py` (model wrapper)

**Risk:** The hand visibility check thresholds (80% both hands, 90% at-least-one) assume Hands23 detections are correct. If Hands23 misses hands in 30% of frames on LATAM videos, the check will systematically reject usable content.

**Priority:** Run Hands23 on LATAM test set with manual ground-truth annotations (50-100 frames, 20 videos minimum). Measure precision/recall at confidence thresholds 0.7, 0.8, 0.9. If recall < 85% at the configured threshold, either (a) lower the threshold, (b) retrain Hands23 on LATAM examples, or (c) switch to a different hand detector.

---

### Face-presence check threshold (0.8 confidence) is not validated (Medium Priority)

**What's not tested:** The face detection confidence threshold of 0.8 (in `pipeline.py:72`) was set to the design spec but is not validated against real LATAM data.

**Files:** `bachman_cortex/checks/face_presence.py`, `bachman_cortex/pipeline.py:72`

**Risk:** SCRFD may have systematic precision/recall characteristics at this threshold. Threshold is too high (misses real faces) or too low (flags false positives) on LATAM videos.

**Test coverage needed:** Run SCRFD on LATAM test set (1000+ frames, varied angles/lighting) with manual ground-truth face annotations. Plot precision-recall curve. Adjust threshold to maximize F1 or to a user-defined operating point (e.g., 90% recall at 95% precision).

---

### Privacy sensitivity is untested (High Priority)

**What's not tested:** Grounding DINO's accuracy at detecting credit cards, ID cards, documents, and screens in LATAM videos.

**Files:** `bachman_cortex/models/grounding_dino_detector.py`, `bachman_cortex/checks/privacy_safety.py` (deleted in recent refactor)

**Risk:** GDINO may have false negatives (misses sensitive objects) or false positives (flags innocuous objects as sensitive). Zero-tolerance privacy check (`check_privacy_safety` requires 0 detections) means any false positive rejects the entire segment. False negatives pass through contaminated data.

**Test coverage needed:** Curate a 200-frame labeled privacy test set (images of credit cards, ID cards, documents, screens from LATAM-style egocentric footage). Measure GDINO precision/recall. If precision < 95%, this check cannot enforce zero-tolerance safely. Either (a) raise `box_threshold` + `text_threshold` to reduce false positives, (b) use YOLO fine-tuned on privacy classes (see YOLOV5N_FINETUNE_PLAN.md), or (c) disable the check.

---

### Luminance / blur thresholds are design-spec only (Medium Priority)

**What's not tested:** The luminance/blur decision table (in `luminance_blur.py`) thresholds for brightness stability, Tenengrad variance, etc. were set per design spec but not validated against real video datasets.

**Files:** `bachman_cortex/checks/luminance_blur.py:50-80` (decision table)

**Risk:** Thresholds may be too strict (rejects usable low-light content) or too lenient (passes blurry footage).

**Test coverage needed:** Collect 100+ LATAM clips with varying lighting (indoor, outdoor, night, overexposed). Manually label which clips are "acceptably bright and sharp" vs. "too dark/blurry". Measure false positive/negative rates at current thresholds. Adjust.

---

*End of concerns audit*
