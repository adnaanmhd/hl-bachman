# Coding Conventions

**Analysis Date:** 2026-04-15

## Naming Patterns

**Files:**
- Module files use `snake_case`: `hand_visibility.py`, `frame_extractor.py`, `segment_ops.py`
- Test files follow naming pattern: `benchmark_*.py` for benchmarking utilities, `generate_*.py` for data generation
- Packages are `lowercase_with_underscores`: `bachman_cortex`, `checks`, `models`, `utils`, `tests`

**Functions:**
- All functions use `snake_case`: `check_hand_visibility()`, `extract_frames()`, `per_frame_to_bad_segments()`
- Helper functions (module-private) prefixed with single underscore: `_hand_in_frame()`, `_nvdec_available()`, `_feature_params()`
- Class methods follow standard Python conventions: `__init__()`, `detect()`, `benchmark()`

**Variables:**
- Local variables use `snake_case`: `total_frames`, `confidence_threshold`, `frame_h`, `frame_w`
- Module-level constants use `UPPER_SNAKE_CASE`: `_BATCH_SIZE = 16`, `_CUDA_LK = False`, `_NVDEC_OK`
- Dictionary keys use `snake_case`: `metric_value`, `confidence`, `frame_dims`

**Types and Classes:**
- Dataclasses use `PascalCase`: `CheckResult`, `TimeSegment`, `FrameLabel`, `CheckableSegment`, `SegmentValidationResult`, `VideoProcessingResult`, `FaceDetection`, `HandDetection`
- Enums use `PascalCase` with `UPPER_SNAKE_CASE` members: `HandSide.LEFT`, `ContactState.NO_CONTACT`

## Code Style

**Formatting:**
- No explicit linter/formatter (pyproject.toml only specifies build system and dependencies)
- Use Python 3.11+ type hints throughout (required: `requires-python = ">=3.11,<3.14"`)
- Line lengths appear flexible (observed up to ~120 characters)
- Indentation: 4 spaces (standard Python)

**Linting:**
- No `.pylintrc`, `.flake8`, or `ruff.toml` detected
- Code follows PEP 8 conventions by observation (snake_case, docstrings, type hints)

## Import Organization

**Order:**
1. Standard library imports: `import os`, `import sys`, `import time`, `import json`, `import dataclasses`, `import cv2`, `import numpy as np`
2. Third-party package imports: `import torch`, `from insightface.app import FaceAnalysis`, `from ultralytics import YOLO`
3. Relative imports from project: `from bachman_cortex.checks.check_results import CheckResult`, `from bachman_cortex.models.hand_detector import HandDetection`

**Path Aliases:**
- No import aliases defined; full paths used: `from bachman_cortex.checks.check_results import CheckResult`
- Absolute imports preferred over relative imports

**Example pattern from `bachman_cortex/checks/hand_visibility.py`:**
```python
import numpy as np

from bachman_cortex.checks.check_results import CheckResult
from bachman_cortex.models.hand_detector import HandDetection, HandSide
```

## Error Handling

**Patterns:**
- **Explicit exception catching**: Check for specific exceptions (`cv2.error`, `ValueError`, `Exception`) before handling
- **Fallback mechanisms**: Try expensive/GPU operations first, fall back to CPU on failure:
  ```python
  if _CUDA_LK:
      try:
          # CUDA path
      except Exception:
          pass  # Fall through to CPU
  ```
- **Early validation**: Check preconditions at function entry, raise `ValueError` with descriptive message:
  ```python
  if not probe.isOpened():
      raise ValueError(f"Cannot open video: {video_path}")
  ```
- **Graceful degradation**: No logging framework; use `print()` for user-facing messages and debug output
- **Return codes**: Check boolean returns from OpenCV operations (`cap.isOpened()`, `cap.grab()`)

**Examples from codebase:**
- `extract_frames()` in `bachman_cortex/utils/frame_extractor.py`: Validates input video exists, falls back CPU when NVDEC unavailable
- `motion_analysis.py`: Try CUDA LK solver, silently use CPU if CUDA unavailable
- `run_batch.py`: Catch exceptions during video processing, continue with next video

## Logging

**Framework:** No logging library imported; **`print()` only**

**Patterns:**
- Informational output to stdout using `print(f"...")` 
- Progress messages during model loading: `print("Loading ML models...")`, `print("  SCRFD loaded")`
- Phase descriptions: `print("\n--- Phase 0: Metadata checks ---")`
- Error/warning output to stderr when needed: `error()` function in `validate.sh` script (not in Python)
- Benchmark/timing output: `print(f"  p50: {result['p50_ms']:.1f}ms | p95: {result['p95_ms']:.1f}ms")`

**When to use print():**
- Loading/initialization messages
- Phase transitions
- Per-video and per-segment results
- Benchmark summaries
- Error conditions (before raising)

**Examples from `bachman_cortex/pipeline.py`:**
```python
print("Loading ML models...")
print(f"  YOLO loaded ({self.config.yolo_model})")
print(f"Extracted {len(frames)} frames ({frame_meta['duration_s']}s video)")
print(f"All models loaded in {elapsed:.1f}s")
```

## Comments

**When to Comment:**
- **Docstrings required** for all public functions and classes
- Comment complex algorithm logic (e.g., optical flow parameters, scoring formulas)
- Document non-obvious parameter meanings or thresholds
- Mark intentional workarounds or heuristics: `# HEURISTIC avoids multi-second EXHAUSTIVE autotune`
- Explain dataset-specific behavior: `# The Hands23 model's left/right classifier can mislabel both hands as the same side`

**DocString/TSDoc:**
- Use **module docstrings** (triple-quoted string at file top) explaining purpose and context:
  ```python
  """Hand visibility check for egocentric video validation.

  Criterion (OR of two conditions):
    - Both hands fully visible in >= 80% of frames, OR
    - At least one hand fully visible in >= 90% of frames.
  """
  ```
- Use **function docstrings** with Args, Returns sections:
  ```python
  def check_hand_visibility(
      per_frame_hands: list[list[HandDetection]],
      frame_dims: tuple[int, int],
  ) -> CheckResult:
      """Check hand visibility using an OR of two sub-conditions.

      Args:
          per_frame_hands: Hand detections per frame.
          frame_dims: (height, width) of each frame.

      Returns:
          CheckResult. ``metric_value`` is the both-hands ratio.
      """
  ```
- Use **inline comments** for parameter ranges and decision logic:
  ```python
  C1 = (0.01 * 255) ** 2  # SSIM stability constant
  C2 = (0.03 * 255) ** 2
  ```
- Class docstrings describe purpose; dataclass fields documented inline:
  ```python
  @dataclass
  class CheckResult:
      """Result of a single acceptance criteria check."""
      status: str  # "pass", "fail", "review", or "skipped"
      metric_value: float  # Raw measurement (e.g., % of frames passing)
  ```

## Function Design

**Size:**
- Prefer shorter functions (< 50 lines for most utility functions)
- Larger functions (100+ lines) reserved for algorithms: `check_motion_combined()`, `extract_frames()`
- Break complex logic into helper functions: `per_frame_to_bad_segments()` calls `_feature_params()`, `_lk_params()`

**Parameters:**
- Use **positional arguments** for essential inputs
- Use **keyword arguments with defaults** for configuration/thresholds:
  ```python
  def check_hand_visibility(
      per_frame_hands: list[list[HandDetection]],
      frame_dims: tuple[int, int],
      confidence_threshold: float = 0.7,
      both_hands_pass_rate: float = 0.80,
      single_hand_pass_rate: float = 0.90,
  ) -> CheckResult:
  ```
- Group related parameters: frame thresholds together, scoring weights together
- Use **dataclass** for optional configuration objects (see `LuminanceBlurConfig`, `PipelineConfig`)

**Return Values:**
- Prefer **single return values**: `CheckResult`, `tuple`, `dict`
- Use **dataclass for multiple related values**: `VideoProcessingResult` bundles 20+ fields
- Return **tuples for fixed pairs**: `extract_frames()` returns `(frames, metadata)`
- Use **dict for flexible key-value data**: Metadata dicts with variable keys

## Module Design

**Exports:**
- Public functions exported at module level: `def check_hand_visibility(...)`
- Private helpers prefixed with `_`: `_hand_in_frame()`, `_nvdec_available()`
- Types (dataclasses, enums) exported alongside functions

**Example from `bachman_cortex/checks/hand_visibility.py`:**
```python
# Private helper (module-only use)
def _hand_in_frame(bbox: np.ndarray, frame_w: int, frame_h: int, margin: int) -> bool:
    """Return True if hand bbox has margin px clearance from frame edges."""

# Public check function (called by pipeline)
def check_hand_visibility(
    per_frame_hands: list[list[HandDetection]],
    ...
) -> CheckResult:
```

**Barrel Files:**
- `__init__.py` files minimal; import model/utility only when needed
- Example from `bachman_cortex/__init__.py`: Only installs cv2 shim, no wildcard imports

**Module Dependencies:**
- `checks/` modules depend on `data_types.py` and `check_results.py` (shared types)
- `models/` modules return detection dataclasses used by checks
- `utils/` modules are leaf nodes (frame_extractor, segment_ops, early_stop have no internal dependencies)
- `pipeline.py` is the orchestrator; imports from all other modules

## Type Hints

**Usage:**
- **Required throughout codebase** (Python 3.11+)
- Use `|` union syntax (not `Union`): `str | Path`, `Optional[int] | None` (use `| None` instead of `Optional`)
- Use `list[T]`, `dict[K, V]`, `tuple[T, ...]` (not `List`, `Dict`, `Tuple`)
- Use `np.ndarray` for numpy arrays with optional dtype hints in comments
- Use `Protocol` for structural typing (duck-typing): `_FrameProcessor` in `frame_extractor.py`

**Examples:**
```python
def check_hand_visibility(
    per_frame_hands: list[list[HandDetection]],
    frame_dims: tuple[int, int],
    confidence_threshold: float = 0.7,
) -> CheckResult:

def extract_frames(
    video_path: str | Path,
    fps: float = 1.0,
    max_frames: Optional[int] = None,
    motion_analyzer: Optional[_FrameProcessor] = None,
) -> tuple[list[np.ndarray], dict]:
```

---

*Convention analysis: 2026-04-15*
