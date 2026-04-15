# Architecture

**Analysis Date:** 2026-04-15

## Pattern Overview

**Overall:** 4-Phase Sequential Pipeline with Early Stopping

The bachman_cortex codebase implements a specialized video validation pipeline that classifies time ranges in egocentric (first-person POV) videos. The pattern is a **sequential, multi-phase pipeline** where each phase produces outputs consumed by the next phase. Phases are orchestrated in `bachman_cortex/pipeline.py` with early-stopping gates (phases 0 and 1) that short-circuit to rejection if criteria fail.

**Key Characteristics:**
- **Phase-based architecture** — Each phase has distinct responsibilities: metadata validation, segment filtering, segment validation, yield calculation
- **Early stopping at gates** — Phase 0 and Phase 1 can reject the entire video before expensive ML inference in Phase 2
- **ML model abstraction** — Models (SCRFD, YOLO, Hands23) are loaded once and reused across frames to avoid redundant inference
- **Cache-driven optimization** — Phase 1 extracts and caches all frames; Phase 2 reuses cached frames, hand detections, and face detections
- **Deterministic check composition** — Each check is a pure function returning a `CheckResult` (status, metric, confidence); multiple checks compose via logical operators

## Layers

**CLI/Orchestration Layer:**
- Purpose: Entry point for batch processing, configuration, multiprocessing coordination
- Location: `bachman_cortex/run_batch.py`
- Contains: Argument parsing, video collection, worker pool management (`multiprocessing.Pool`)
- Depends on: `ValidationProcessingPipeline`, reporting
- Used by: User runs `validate.sh` → `hl-validate` → `python -m bachman_cortex.run_batch`

**Pipeline Layer:**
- Purpose: Implements 4-phase validation logic, orchestrates models and checks
- Location: `bachman_cortex/pipeline.py`
- Contains: `ValidationProcessingPipeline` class, phase methods (`_phase0_metadata`, `_phase1_segment_filter`, `_phase2_segment_validation`, `_phase3_yield`)
- Depends on: Models, checks, utilities (frame extractor, segment ops, transcode)
- Used by: `run_batch.py` instantiates one per worker

**Model Layer:**
- Purpose: Wraps ML models for consistent detection interface
- Locations: `bachman_cortex/models/` (scrfd_detector.py, yolo_detector.py, hand_detector.py)
- Contains: Model wrappers returning typed Detection objects (FaceDetection, Detection, HandDetection)
- Depends on: Third-party libraries (insightface, ultralytics, ONNX Runtime)
- Used by: Pipeline loads and calls `.detect()` / `.detect_batch()` methods

**Check Layer:**
- Purpose: Acceptance criteria as pure functions returning CheckResult
- Location: `bachman_cortex/checks/`
- Contains: Individual check functions per criterion (luminance_blur, hand_visibility, motion_analysis, etc.)
- Depends on: Model detection types, CheckResult dataclass
- Used by: Pipeline calls checks at specific phases

**Utility Layer:**
- Purpose: Shared algorithms for frame extraction, segment operations, motion analysis
- Location: `bachman_cortex/utils/`
- Contains: Frame extraction (with optional NVDEC), segment merging/filtering, motion analysis, transcoding
- Depends on: OpenCV, numpy
- Used by: Pipeline coordinates these utilities

**Data Type Layer:**
- Purpose: Shared dataclass definitions for type safety
- Location: `bachman_cortex/data_types.py`
- Contains: TimeSegment, FrameLabel, CheckFrameResults, CheckableSegment, SegmentValidationResult, VideoProcessingResult
- Depends on: CheckResult from checks module
- Used by: All layers

**Reporting Layer:**
- Purpose: Convert processing results to human-readable reports (Markdown, JSON)
- Location: `bachman_cortex/reporting.py`
- Contains: Timeline entry construction, per-video report formatting, batch report aggregation
- Depends on: CheckResult, VideoProcessingResult
- Used by: `run_batch.py` calls after each video completes

## Data Flow

**Batch Initialization:**
1. `run_batch.py` parses arguments and collects video paths
2. Creates `PipelineConfig` from CLI args
3. Spawns worker pool (if parallel) or uses single sequential pipeline
4. Each worker calls `pipeline.load_models()` once (SCRFD, YOLO, Hands23)

**Per-Video Processing:**

1. **Phase 0: Metadata Gate**
   - Input: Video file path
   - Process: `ffprobe` extracts metadata; `run_all_metadata_checks()` validates format, codec, resolution, FPS, duration, orientation
   - Output: `metadata` dict, `meta_results` dict[str, CheckResult]
   - Decision: If ANY check fails → return VideoProcessingResult with metadata_passed=False, zero usable duration

2. **Single-Pass Frame Extraction (between Phase 0 & 1)**
   - Input: Video path, sampling FPS (e.g., 1 fps), optional MotionAnalyzer
   - Process: `extract_frames()` opens video with NVDEC (CUDA) or cv2.VideoCapture; samples frames; tees native-rate stream to MotionAnalyzer
   - Output: frames list, frame_meta dict, populated motion_analyzer
   - Why: MotionAnalyzer processes full-resolution native frames once; Phase 2 reuses its statistics without re-opening video

3. **Phase 1: Segment Filtering**
   - Input: frames (downscaled to 720p long-edge), frame metadata
   - Process:
     - `_phase1_run_inference()` batches frames through:
       - YOLO detector (batched) for person detection
       - SCRFD detector (ThreadPool) for face detection
       - Hands23 detector (sequential) for hand detection
     - `eval_face_presence()` marks frames with confident faces as bad (privacy gate)
     - `eval_participants()` marks frames with other persons as bad
     - `per_frame_to_bad_segments()` converts per-frame labels to time segments
     - `merge_bad_segments()` unions overlapping bad segments across checks
     - `compute_good_segments()` inverts bad segments
     - `filter_checkable_segments()` keeps only segments ≥ min_checkable_segment_sec (e.g., 60s)
   - Output: prefiltered_segments (CheckableSegment list), phase1_cache (frames + detections)
   - Decision: If no segments remain → return with zero usable duration

4. **Phase 2: Segment Validation**
   - Input: prefiltered_segments, phase1_cache (frames, detections), motion_analyzer
   - Process: For each CheckableSegment:
     - Slice cached frames and detections to segment range
     - Run parallel motion check (camera stability, frozen segments) via ThreadPoolExecutor
     - Run sequential checks:
       - `check_luminance_blur()` — brightness stability + blur detection
       - `check_hand_visibility()` — both-hands ≥80% OR single-hand ≥90%
       - `check_hand_object_interaction()` — hand pose keypoints in interacting state
       - `check_view_obstruction()` — dark/obstructed frame ratio
       - `check_pov_hand_angle()` — hand angle relative to POV
       - `check_face_presence()` — strict: zero-tolerance for faces
     - Collect failing_checks list; segment passes if all checks pass
   - Output: SegmentValidationResult list (segment + pass/fail + per-check results)
   - Cache reuse: All detections are from Phase 1; no new ML inference

5. **Phase 3: Yield Calculation**
   - Input: segment_results, original_duration_sec
   - Process: usable_sec = sum of passed segments; unusable_sec = original - usable; yield = usable / original
   - Output: usable_duration_sec, unusable_duration_sec, yield_ratio

**Reporting:**
- `write_video_report()` converts SegmentValidationResult list to timeline (HH:MM:SS.mmm ranges + reasons)
- `write_batch_report()` aggregates per-video results and computes batch statistics
- JSON serialized via `dataclasses.asdict()` for machine-readable output

**State Management:**

- **Per-worker state**: Single `ValidationProcessingPipeline` instance with models loaded once. Models are thread-safe (ONNX Runtime, InsightFace use thread pools internally).
- **Per-video state**: `VideoProcessingResult` accumulates data through all phases; immutable after construction
- **Phase 1 cache**: `phase1_cache` dict holds frames, detections, dimensions; passed to Phase 2 and then freed after Phase 2 completes
- **Motion state**: `MotionAnalyzer` accumulates statistics during single-pass decode; frozen copy passed to Phase 2

## Key Abstractions

**CheckResult:**
- Purpose: Uniform interface for all checks to report status, metric, confidence, and debug details
- Examples: `bachman_cortex/checks/check_results.py`
- Pattern: All checks return `CheckResult(status, metric_value, confidence, details)`

**Detection Objects:**
- Purpose: Typed wrappers for model outputs to avoid magic tuples
- Examples: `FaceDetection` (bbox, confidence, landmarks), `Detection` (bbox, confidence, class_id, class_name), `HandDetection` (bbox, confidence, side, keypoints)
- Used by: Checks unpack these to evaluate criteria

**TimeSegment & FrameLabel:**
- Purpose: Represent time ranges and per-frame decisions
- Examples: `TimeSegment(start_sec, end_sec)`, `FrameLabel(frame_idx, timestamp_sec, passed, confidence, labels)`
- Pattern: Per-frame labels → bad segments → good segments → CheckableSegment

**CheckableSegment:**
- Purpose: A pre-filtered good segment eligible for Phase 2 validation
- Fields: segment_idx, start_sec, end_sec, duration property
- Lifecycle: Created in Phase 1, validated in Phase 2, reported in Phase 3

**SegmentValidationResult:**
- Purpose: Aggregate Phase 2 check results for one segment
- Fields: segment, passed (boolean), check_results (dict[str, CheckResult]), failing_checks (list[str])
- Used by: Reporting to explain why segment passed/failed

**VideoProcessingResult:**
- Purpose: Complete immutable output of one video
- Fields: metadata, phase0 results, phase1 results, phase2 results, phase3 metrics
- Serialized: JSON via `dataclasses.asdict()`

**MotionAnalyzer:**
- Purpose: Single-pass camera stability and frozen segment detection
- Pattern: Stateful during frame decode (`process_frame()`), read-only during Phase 2
- Used by: Phase 2 calls `check_motion_combined_from_analyzer()` without re-decoding

## Entry Points

**`validate.sh`:**
- Location: `/home/egocentric-humynlabs/Documents/hl-bachman/validate.sh`
- Triggers: User runs `./validate.sh /path/to/videos/`
- Responsibilities: Check Python version, FFmpeg, git; create venv; install deps; download models; call `hl-validate`

**`hl-validate` (via pyproject.toml):**
- Location: Entry point defined in `pyproject.toml` → calls `run_batch.main()`
- Triggers: After `pip install -e .`
- Responsibilities: Alias for `python -m bachman_cortex.run_batch`

**`python -m bachman_cortex.run_batch`:**
- Location: `bachman_cortex/run_batch.py`
- Triggers: CLI with video paths and optional flags (--fps, --workers, --output, etc.)
- Responsibilities:
  - Collect video files
  - Auto-detect worker count or use --workers flag
  - Sequential: Single pipeline instance, models loaded once
  - Parallel: multiprocessing.Pool, each worker calls `_init_worker()` to load models
  - Write per-video reports and batch report
  - Update run index

**Pipeline Direct Use (for testing/scripting):**
```python
from bachman_cortex.pipeline import ValidationProcessingPipeline, PipelineConfig
config = PipelineConfig(sampling_fps=1.0)
pipeline = ValidationProcessingPipeline(config)
pipeline.load_models()
result = pipeline.process_video("video.mp4", "output_dir/")
```

## Error Handling

**Strategy:** Fail-fast at early gates; graceful degradation for isolated segment/check failures

**Patterns:**

- **Metadata gate fails** (`Phase 0`): Entire video rejected; no ML inference runs; yield = 0%
- **No checkable segments** (`Phase 1`): Entire video rejected; yield = 0% (metadata passed but no usable content)
- **Phase 2 check fails**: Individual segment fails; contributing to unusable duration; does not block other segments
- **Model loading fails**: Pipeline detects unloaded state; lazy-loads on first video if not pre-loaded
- **Transcoding error**: Caught in `maybe_transcode_hevc_to_h264()`; falls back to original path
- **Worker exception** (parallel): Traceback printed; error_result added to batch; processing continues

**Try-Catch Locations:**
- `run_batch.py:_process_video_worker()` — Wraps entire video processing; returns error_result if exception
- `pipeline.py:_warmup_models()` — Individual model warmup failures logged but don't halt
- `extract_frames()` — Validates video can be opened; raises ValueError if not

## Cross-Cutting Concerns

**Logging:** 
- Approach: `print()` statements to stdout/stderr; no logging framework configured
- Pattern: Progress messages at phase boundaries, per-check results formatted as "check_name status metric=X.XXXX"
- Examples: Phase 0/1/2/3 headings; frame count progress; inference timing

**Validation:**
- Metadata checks: Deterministic rules (format, codec, resolution, FPS, duration, orientation)
- ML checks: Confidence-based thresholds on detection counts, pass rates, angles
- Segment checks: Duration thresholds (min_checkable_segment_sec, min_bad_segment_sec)

**Authentication:** 
- Not applicable — no external API calls; local file processing only

**Configuration:**
- `PipelineConfig` dataclass: ~40 parameters covering model paths, thresholds, processing options
- CLI flags in `run_batch.py` expose critical knobs (--fps, --min-segment, --workers, --hevc-to-h264)
- Default values tuned for egocentric hand pose dataset (720p downscaling, 80%/90% hand visibility, 60s min segment)

---

*Architecture analysis: 2026-04-15*
