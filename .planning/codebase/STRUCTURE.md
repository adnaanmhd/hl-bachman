# Codebase Structure

**Analysis Date:** 2026-04-15

## Directory Layout

```
hl-bachman/
├── bachman_cortex/               # Main package (Python module)
│   ├── __init__.py               # Installs cv2.dnn shim before imports
│   ├── _cv2_dnn_shim.py          # Patches cv2.dnn.blobFromImage for custom CV2 builds
│   ├── data_types.py             # Shared dataclasses (TimeSegment, CheckResult, etc.)
│   ├── pipeline.py               # 4-phase validation pipeline orchestrator
│   ├── run_batch.py              # CLI batch runner, multiprocessing coordinator
│   ├── reporting.py              # Report generation (Markdown, JSON)
│   ├── checks/                   # Per-check acceptance criteria functions
│   │   ├── check_results.py      # CheckResult dataclass definition
│   │   ├── video_metadata.py     # Phase 0: Format, codec, resolution, FPS, duration, orientation
│   │   ├── face_presence.py      # Phase 2: Strict face detection (zero tolerance)
│   │   ├── hand_visibility.py    # Phase 2: Both-hands ≥80% OR single-hand ≥90%
│   │   ├── hand_object_interaction.py  # Phase 2: Hand pose keypoints in interaction state
│   │   ├── view_obstruction.py   # Phase 2: Dark/obstructed frame ratio
│   │   ├── pov_hand_angle.py     # Phase 2: Hand angle relative to POV
│   │   ├── motion_analysis.py    # Phase 2: Camera stability + frozen segments (40KB)
│   │   ├── participants.py       # Phase 1: Other person detection via YOLO + body part heuristic
│   │   └── luminance_blur.py     # Phase 2: Brightness stability + blur detection
│   ├── models/                   # ML model wrappers
│   │   ├── scrfd_detector.py     # SCRFD face detection via InsightFace
│   │   ├── yolo_detector.py      # YOLO11 object detection (persons)
│   │   ├── hand_detector.py      # Hands23 hand detection (ONNX Runtime)
│   │   ├── hand_detector_100doh.py  # Alternative hand detector (legacy)
│   │   ├── download_models.py    # Model weight downloader
│   │   └── weights/              # Model files directory (git-ignored; ~1.2GB on disk)
│   │       ├── insightface/      # SCRFD buffalo_sc model
│   │       └── hands23_detector/ # Hands23 repo (ONNX model + inference code)
│   ├── utils/                    # Shared utilities
│   │   ├── frame_extractor.py    # Video frame extraction (NVDEC or cv2.VideoCapture)
│   │   ├── segment_ops.py        # Segment merging/filtering (per-frame to time segments)
│   │   ├── early_stop.py         # Per-frame evaluation helpers (face presence, participants)
│   │   ├── motion_analysis.py    # MotionAnalyzer class (stateful frame processor)
│   │   ├── video_metadata.py     # Metadata extraction via ffprobe
│   │   └── transcode.py          # HEVC → H.264 lossless transcoding
│   └── tests/                    # Benchmarks and test utilities (no pytest fixtures)
│       ├── benchmark_models.py   # Inference timing benchmarks
│       ├── benchmark_phase_correlation.py  # Motion analysis benchmarking
│       └── generate_test_video.py # Test video generation utility
├── .planning/                    # GSD planning output
│   └── codebase/                 # This analysis (ARCHITECTURE.md, STRUCTURE.md)
├── .claude/                      # Claude IDE metadata
├── scripts/                      # Auxiliary scripts (transcoding, etc.)
├── validate.sh                   # Main entry point: setup + run pipeline
├── pyproject.toml                # Package metadata, entry points, dependencies
├── README.md                     # User documentation, architecture diagram
├── checks.md                     # Detailed check criteria documentation
├── .gitignore                    # Excludes .venv, logs/, vid_samples/, model weights
└── (other planning docs)         # OPTIMIZATION_PLAN*.md, idea-brief.md, etc.
```

## Directory Purposes

**`bachman_cortex/`:**
- Purpose: Main Python package for video validation
- Contains: Pipeline orchestrator, checks, models, utilities, reporting
- Key files: `pipeline.py` (757 lines), `run_batch.py` (379 lines), `reporting.py` (486 lines)

**`bachman_cortex/checks/`:**
- Purpose: Modular acceptance criteria checks (one check = one file pattern)
- Contains: 10 active checks + 1 shared result type
- Check pattern: `def check_NAME(inputs..., **kwargs) -> CheckResult: ...`
- Notable: `motion_analysis.py` is 40KB (40K lines) due to detailed camera stability math

**`bachman_cortex/models/`:**
- Purpose: ML model wrappers with consistent interfaces
- Contains: SCRFD (face), YOLO (person), Hands23 (hand detection)
- Pattern: Each model has `.detect(frame)` and optionally `.detect_batch(frames)` methods
- Dependencies: InsightFace (SCRFD), Ultralytics (YOLO), ONNX Runtime (Hands23)

**`bachman_cortex/utils/`:**
- Purpose: Shared algorithms not tied to specific checks
- Contains: Frame extraction, segment operations, motion analysis, video metadata, transcoding
- Key abstractions: `MotionAnalyzer` (stateful), `TimeSegment`, `FrameLabel`

**`bachman_cortex/tests/`:**
- Purpose: Benchmarking and test data generation (no unit test framework)
- Contains: Performance measurement scripts for models and motion analysis
- Run manually: `python -m bachman_cortex.tests.benchmark_models`

## Key File Locations

**Entry Points:**
- `validate.sh`: Bash wrapper for environment setup and pipeline launch
- `run_batch.py`: Main CLI entry point; handles arguments, worker pool, batch reporting
- `pipeline.py`: Core 4-phase orchestrator; called once per video per worker

**Configuration:**
- `pyproject.toml`: Package metadata, dependencies, entry point `hl-validate`
- `PipelineConfig` class in `pipeline.py` (lines 59–132): ~40 tunable parameters

**Core Logic:**
- `pipeline.py`: `ValidationProcessingPipeline` class with phase methods (757 lines total)
- `checks/*.py`: Individual check functions composing via `OR`/`AND` operators
- `models/*.py`: Model initialization and inference wrappers
- `utils/segment_ops.py`: Bad/good segment conversion and merging

**Testing:**
- `tests/benchmark_models.py`: Measure inference speed of SCRFD, YOLO, Hands23
- `tests/benchmark_phase_correlation.py`: Profile motion analyzer performance
- No pytest fixtures; tests are standalone scripts

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `hand_visibility.py`, `frame_extractor.py`)
- Test/benchmark files: `benchmark_NAME.py`, `generate_NAME.py`
- Entry points: `run_batch.py`, `validate.sh`

**Directories:**
- Package: `lowercase` (e.g., `bachman_cortex`, `checks`, `models`, `utils`)
- Logical groups: `checks/`, `models/`, `utils/`, `tests/`

**Functions:**
- Checks: `check_NAME(inputs, **config) -> CheckResult` (e.g., `check_hand_visibility`, `check_luminance_blur`)
- Helpers: `_name_helper()` (single underscore = module-private)
- Entry point: `main()` in `run_batch.py`

**Classes:**
- Models: `NAMEDetector` (e.g., `SCRFDDetector`, `YOLODetector`, `HandObjectDetectorHands23`)
- Pipeline: `ValidationProcessingPipeline`
- Analyzers: `MotionAnalyzer`
- Results: `CheckResult`, `VideoProcessingResult`, `SegmentValidationResult`

**Dataclasses:**
- Detection types: `FaceDetection`, `Detection`, `HandDetection`
- Time/segment: `TimeSegment`, `FrameLabel`, `CheckableSegment`
- Aggregates: `CheckFrameResults`, `SegmentValidationResult`, `VideoProcessingResult`

## Where to Add New Code

**New Check:**
1. Create `bachman_cortex/checks/check_NAME.py`
2. Import `CheckResult` from `checks/check_results.py`
3. Implement `def check_NAME(inputs, **config) -> CheckResult:`
4. Call from `pipeline.py` at appropriate phase (1 or 2)
5. Add CLI arg in `run_batch.py` if configurable threshold needed
6. Document in `checks.md`

Example location: `bachman_cortex/checks/check_new_criterion.py`

**New Model:**
1. Create `bachman_cortex/models/NAME_detector.py`
2. Define detection dataclass (inherit pattern from `FaceDetection`)
3. Implement model wrapper with `.detect()` and optional `.detect_batch()`
4. Add initialization to `pipeline.py:load_models()` (line 141–159)
5. Call from appropriate phase check

Example location: `bachman_cortex/models/depth_detector.py`

**New Utility:**
1. Create `bachman_cortex/utils/utility_name.py` if logic is reusable
2. Or inline in `pipeline.py` if single-use
3. Follow module-private naming (`_helper()`) for internals

Example location: `bachman_cortex/utils/color_balance.py`

**Configuration Parameter:**
1. Add field to `PipelineConfig` dataclass in `pipeline.py` (lines 59–132)
2. Set sensible default value
3. Add CLI arg in `run_batch.py:main()` if user-facing (lines 139–160)
4. Document in docstring and README

**Test/Benchmark:**
1. Create `bachman_cortex/tests/test_NAME.py` or `benchmark_NAME.py`
2. No pytest fixture; run as `python -m bachman_cortex.tests.test_NAME`
3. Print results to stdout

Example: `python -m bachman_cortex.tests.benchmark_new_model`

## Special Directories

**`bachman_cortex/models/weights/`:**
- Purpose: Downloaded model files (~1.2GB total)
- Generated: Yes (via `download_models.py` during setup)
- Committed: No (listed in `.gitignore`)
- Content: InsightFace SCRFD, Hands23 ONNX models
- Note: Lazy-loaded; files are not required at install time, only at runtime

**`.planning/codebase/`:**
- Purpose: GSD codebase analysis documents
- Generated: Yes (by codebase mapper agent)
- Committed: Yes
- Contains: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md
- Note: Updated by `/gsd-map-codebase` command

**`logs/`:**
- Purpose: Runtime logs from pipeline execution
- Generated: Yes (optional, when debugging)
- Committed: No (listed in `.gitignore`)
- Content: Text logs of per-video processing

**`vid_samples/`:**
- Purpose: Sample videos for testing
- Generated: No (user-provided or generated by `generate_test_video.py`)
- Committed: No (listed in `.gitignore`)

---

*Structure analysis: 2026-04-15*
