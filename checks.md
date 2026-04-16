# Validation & Processing Pipeline — Check Specifications

The pipeline operates in 4 phases to classify every time range in an
egocentric video as **USABLE**, **UNUSABLE**, or **REJECTED**. No clip
files are produced — the output is a timestamped report.

## Phase 0: Video Metadata (gate — failure rejects entire video)

| Check       | Acceptance Condition                           | How It Is Estimated              |
| ----------- | ---------------------------------------------- | -------------------------------- |
| Format      | MP4 container (MPEG-4)                         | Container metadata via FFprobe   |
| Encoding    | H.264 or HEVC (H.265) video codec              | Codec metadata via FFprobe       |
| Resolution  | Displayed dims >= 1920 x 1080 (rotation applied)                    | Width & height metadata          |
| Frame Rate  | >= 28 FPS                                                           | FPS metadata                     |
| Duration    | >= 119 seconds                                                      | Duration metadata                |
| Orientation | Rotation in {0, 90, 270} and displayed width > displayed height     | Width, height, rotation metadata |

## Phase 1: Segment Filtering (face, participants)

Phase 1 scans ALL frames (no early stopping) to identify bad segments. Bad
segments from both checks are merged (overlapping regions unified), and
the remaining good segments are filtered by minimum duration (default 60s,
configurable).

| Check         | Per-Frame Pass Condition                                       | Model           |
| ------------- | -------------------------------------------------------------- | --------------- |
| Face Presence | No face detection with confidence >= 0.8                       | SCRFD-2.5GF     |
| Participants  | Zero other persons detected (wearer's body parts filtered out) | YOLO11s + SCRFD |

### Segment Logic

- **Bad segment:** Contiguous frames that fail any Phase 1 check.
- **1-second granularity:** Each frame at timestamp T covers \[T, T+1) seconds. A single bad frame = 1-second bad segment.
- **Forgiveness threshold:** Bad segments with duration <= 2 seconds (configurable via `--min-bad-segment`) are forgiven and treated as good. Segments with duration > 2s are kept. Applied per-check before merging, and again after merging. This prevents isolated single-frame detections from fragmenting the video.
- **Overlap merging:** Remaining bad segments from both checks are unioned. E.g., face bad \[85s, 89s) + participant bad \[87s, 91s) → merged \[85s, 91s).
- **Good segments:** The inverse of unified bad segments over \[0, total\_duration].
- **Checkable segments:** Good segments >= minimum duration threshold (default 60s, configurable via `--min-segment`). These advance to Phase 2.
- **Discarded segments:** Good segments below the minimum duration threshold. Labeled **UNUSABLE** with reason `segment_too_short` in the timeline (they are not re-validated in Phase 2 because pass-rate metrics are unreliable on <60 frames).
- **No padding:** Segment boundaries are reported exactly at detected frame timestamps.

## Phase 2: Segment Validation (pass/fail per checkable segment)

Each checkable segment from Phase 1 is validated against all remaining checks.
Phase 2 is binary pass/fail per segment. Segments that pass all checks are
labeled **USABLE** in the timeline; segments that fail any check are labeled
**REJECTED** with the failing check names and observed metrics.

| Check                   | Acceptance Condition                                                                          | Model / Method                                  |
| ----------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Luminance               | >= 80% good frames (no dead black / too dark / blown out / flicker)                            | Mean grayscale + rolling stddev flicker          |
| Pixelation              | >= 80% non-pixelated frames (blockiness ratio <= 1.5)                                          | Block boundary gradient ratio (8x8 grid)        |
| Camera Stability        | High-pass filtered LK jitter score <= 0.181                                                   | Sparse Lucas-Kanade optical flow at 0.5x, high-pass filtered (GPU when available) |
| Frozen Segments         | No > 30 consecutive frames with near-zero LK motion (trans < 0.1px, rot < 0.001°)             | LK optical flow signal (zero marginal cost)     |
| Hand Visibility         | >= 80% frames with both hands fully in frame **OR** >= 90% frames with at least one hand fully in frame (bbox > 0 px from every edge, confidence >= 0.7) | Hands23 detection + bbox edge clearance         |
| Hand-Object Interaction | Hand contact with portable or stationary object in >= 60% frames                              | Hands23 contact state                           |
| View Obstruction        | <= 10% frames obstructed                                                                      | OpenCV heuristics                               |
| POV-Hand Angle          | Hand angle from frame center < 40° in >= 60% frames                                           | Hand bbox center vs frame center                |
| Face Presence (strict)  | Zero sampled frames in the segment contain a face with confidence >= 0.8                       | SCRFD-2.5GF (detections reused from Phase 1 cache) |

### Camera Stability — Single-Pass LK with High-Pass Jitter Filter

LK optical flow is computed **inline with frame extraction** by a
stateful `MotionAnalyzer` (`bachman_cortex/checks/motion_analysis.py`).
The video is decoded exactly once (via NVDEC when `cv2.cudacodec` is
available, else `cv2.VideoCapture`); at every native frame whose index
is a multiple of `frame_skip = round(native_fps / target_fps)` the
analyzer:

1. Converts to grayscale and downsamples by `fast_scale=0.5` (~360p
   from 720p input).
2. Tracks corner features frame-to-frame with sparse Lucas-Kanade (CUDA
   `cv2.cuda.SparsePyrLKOpticalFlow` when available; CPU fallback otherwise).
3. Estimates an affine transform via RANSAC, yielding per-frame
   translation (px) and rotation (degrees).

Phase 2's stability + frozen-segment check then slices the segment's range
from the analyzer (no `cv2.VideoCapture.set()` seek, no re-decode).

**High-pass jitter filter:** Before scoring, the full per-frame translation
and rotation timeseries are high-pass filtered to separate intentional camera
movement (smooth pans, head turns) from unwanted jitter (shake, vibration).
A rolling mean over a configurable window (default 0.5s = 15 frames at 30 FPS)
approximates the intended motion; the absolute residual is the jitter signal.
Only the jitter is scored.

Per-second jitter scores are computed using weighted components:

| Component              | Weight | Normalisation                     |
| ---------------------- | ------ | --------------------------------- |
| Mean translation       | 0.35   | avg_t / (trans_threshold \* 3)    |
| Translation variance   | 0.25   | std_t / (variance_threshold \* 2) |
| Mean rotation          | 0.20   | avg_r / (rot_threshold \* 3)      |
| Sudden jumps (> 30 px) | 0.20   | (jump_count / n) \* 10            |

All component scores are clamped to [0, 1]; the weighted sum produces a
per-second score in [0, 1].

**Final verdict:** The overall score is the mean of all per-second scores.
The segment passes if overall score <= `shaky_score_threshold` (default 0.181).

CUDA GPU acceleration is used automatically when OpenCV is built with CUDA
support (`cv2.cuda.SparsePyrLKOpticalFlow`), with transparent CPU fallback.

### Luminance

Per-frame luminance classification with brightness flicker detection.
Pass if >= 80% of frames are good (no luminance reject and no flicker).

| Condition   | Mean Luminance | Decision |
| ----------- | -------------- | -------- |
| Dead black  | < 15           | Reject   |
| Too dark    | 15 - 45        | Reject   |
| Usable      | 45 - 230       | Accept   |
| Blown out   | > 230          | Reject   |

**Flicker detection:** Rolling stddev over a 10-frame window. If stddev > 30,
all frames in the window are rejected as `brightness_flicker`.

### Pixelation

Block boundary gradient ratio at 8x8 grid. Measures the ratio of gradient
energy at block boundaries vs. interior positions. Compression artifacts and
upscaled low-res sources produce artificially strong edges at block boundaries.

Pass if >= 80% of frames have blockiness ratio <= 1.5. Clean footage scores
~0.97-1.0; pixelated footage scores >> 1.5.

### Face Presence (strict)

Rejects any segment whose sampled frames contain even a single face detected
at or above `face_confidence_threshold` (0.80). This is stricter than the
Phase 1 face-presence gate — Phase 1 discards only contiguous bad runs longer
than `--min-bad-segment` seconds (default 2s), so brief flashes of a face can
survive into checkable segments. Phase 2 face presence re-checks with zero
tolerance.

Reuses SCRFD detections cached from Phase 1 — no extra inference. The metric
reported is the fraction of frames with **no** prominent face (higher is
better, 1.0 = clean). Details include `frames_with_prominent_face`,
`total_frames`, and `max_face_confidence_seen`.

### POV-Hand Angle

Measures whether detected hands fall within a plausible first-person (egocentric)
field of view. For each hand bounding box, the center is computed and its pixel
distance from the frame center is normalized by the frame's half-diagonal. This
normalized distance is mapped to an angle using an assumed diagonal FOV of 90°:

    angle = (pixel_dist / half_diagonal) * (diagonal_fov / 2)

A frame passes if ALL detected hands have angle < 40°. Frames with no hands
detected count as failures. The segment passes if >= 60% of frames pass.

## Phase 3: Yield Calculation

| Metric             | Definition                                                                                         |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| Usable footage     | Sum of durations of all segments that passed Phase 2                                                |
| Unusable footage   | Phase 1 bad segments + Phase 1 discarded short segments (< min duration) + Phase 2 rejected segments |
| Yield              | Usable footage / original video duration                                                            |

**Invariant:** Original duration = usable footage + unusable footage.

## Output

The pipeline produces timestamps and reports only — no clip files are cut
from the source video. Each range of the source is categorized as one of:

| Category       | Meaning                                                                        |
| -------------- | ------------------------------------------------------------------------------ |
| **USABLE**     | Passed both Phase 1 filtering and all enabled Phase 2 validation checks.        |
| **UNUSABLE**   | Filtered in Phase 1 (face / participants / privacy), or a clean gap too short to validate (`segment_too_short`). |
| **REJECTED**   | Checkable segment that ran through Phase 2 but failed one or more validation checks. Also: the single span of a video rejected at the Phase 0 metadata gate. |

### Per Video

- **`report.md`** — A markdown report whose core artifact is a unified
  chronological timeline in `HH:MM:SS.mmm` format. Every row is one
  contiguous range of the source video, labeled with its category and,
  for non-USABLE ranges, the failing check names plus the observed
  metric vs. the accepted threshold (e.g., `ml_hand_visibility (accepted
  both>=80% OR single>=90%; observed both=42%, single=58%)`). Includes
  the Phase 0 metadata table. For metadata-fail videos the timeline
  collapses to a single REJECTED row covering the whole video.
- **`{video_name}.json`** — Machine-readable dump of the complete
  `VideoProcessingResult` dataclass: all segments, per-check results,
  metadata, and timings.

### Per Batch

- **`batch_report.md`** — Topline stats: total yield, total usable / unusable durations, per-video summary table.
- **`batch_results.json`** — Machine-readable batch results.

### Output Directory Structure

```
results/run_NNN/
├── {video_name}/
│   ├── report.md
│   └── {video_name}.json
├── batch_report.md
└── batch_results.json
```

## Performance Optimizations

### Video Decode

- **Single-pass decode:** Frame extraction, 720p downscale, and Phase 2
  motion analysis share one pass over the video.
  `frame_extractor.extract_frames` decodes each frame, resizes it to
  720p long-edge, feeds the resized frame to a `MotionAnalyzer` (LK
  accumulator), and appends it to the output list. Phase 2's stability +
  frozen-segment check slices the pre-computed per-second data without
  re-decoding.
- **NVDEC hardware decode:** When the cv2 build has `cudacodec`,
  `extract_frames` uses `cv2.cudacodec.VideoReader` for GPU-side H.264
  decode. Frames are downloaded to CPU (BGR numpy) only at sample /
  motion points. Falls back to `cv2.VideoCapture` when cudacodec is
  absent. Output frames are identical in both paths.

### Phase 1

- **No early stopping:** All frames are scanned to identify segment boundaries.
- **720p long-edge downscale at extraction:** Each sampled frame is
  resized to 720p long-edge during `extract_frames` before being stored
  in memory (`resize_long_edge` parameter, default
  `PipelineConfig.phase1_long_edge = 720`). Full-resolution frames never
  accumulate — peak frame memory is N × 2.7 MB (720p) instead of
  N × 6.2 MB (1080p). YOLO / SCRFD / Hands23 all see the downscaled
  frames; bounding boxes are in 720p coordinates.
- **Threaded SCRFD:** Face detection is dispatched across the batch via
  a `ThreadPoolExecutor` (`PipelineConfig.scrfd_threads`, default 4),
  overlapping with YOLO and Hands23 in the inner loop.
- **Model pre-warm + cuDNN HEURISTIC:** `load_models()` runs one
  synthetic-frame forward pass per model; SCRFD uses
  `cudnn_conv_algo_search=HEURISTIC` to skip ORT's multi-second
  EXHAUSTIVE autotune on first inference.
- **Batched YOLO inference:** Object detection processes 16 frames per batch.

### Phase 2

- **Motion results derived, not recomputed:** Camera stability and
  frozen-segment checks slice the per-second data from the
  `MotionAnalyzer` populated during extraction. No per-segment
  `cv2.VideoCapture` re-open, no redundant decode, no LK re-compute.
- **Cached detections reused:** Hand / face detections from Phase 1 are
  sliced by segment range; no extra ML inference runs in Phase 2.
- **Segments processed sequentially:** GPU models are shared; sequential processing avoids GPU contention.

### Hands23 Input Downscaling

The Hands23 detector (Faster R-CNN X-101-FPN) is the most compute-intensive model
in the pipeline. Frames are downscaled before inference — long edge capped at
`hands23_max_resolution` (default 720), preserving aspect ratio. Output bounding box
coordinates are automatically scaled back to original dimensions.

Redundant with the Phase 1 720p downscale above, but kept as a safety net
in case a caller sets `phase1_long_edge=None` (native resolution).

### Camera Stability FPS Cap

Caps optical flow analysis at 30 FPS regardless of native frame rate.

### Frozen Segment Subsampling

Samples at 10 FPS instead of native FPS for frozen detection.
