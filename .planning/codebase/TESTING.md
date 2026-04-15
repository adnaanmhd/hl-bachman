# Testing Patterns

**Analysis Date:** 2026-04-15

## Test Framework

**Runner:**
- **No pytest/unittest framework detected** — tests are standalone scripts
- Test files located in `bachman_cortex/tests/`
- Tests are **executable Python modules**, not pytest suites

**Assertion Library:**
- No standard assertion library used (no pytest, unittest, or nose)
- Tests use **print output for validation** and manual inspection
- Benchmarks validate latency percentiles and detection counts via printed output

**Run Commands:**
```bash
# Run benchmark suite for all 4 ML models
python -m bachman_cortex.tests.benchmark_models

# Run phase-correlation motion analysis benchmark
python -m bachman_cortex.tests.benchmark_phase_correlation

# Generate test video for benchmarking
python -m bachman_cortex.tests.generate_test_video
```

## Test File Organization

**Location:**
- All test/benchmark files co-located in `bachman_cortex/tests/` directory
- Not separated from source code; integrated into package

**Naming:**
- Benchmark scripts: `benchmark_*.py` (e.g., `benchmark_models.py`, `benchmark_phase_correlation.py`)
- Utility/generation scripts: `generate_*.py` (e.g., `generate_test_video.py`)
- Init file: `__init__.py` (empty, allows directory to be a package)

**File Structure:**
```
bachman_cortex/tests/
├── __init__.py                           # Empty
├── benchmark_models.py                   # Benchmark all 4 ML detectors
├── benchmark_phase_correlation.py        # Compare motion analysis approaches
└── generate_test_video.py                # Create synthetic test video
```

## Test Structure

**Suite Organization:**
Tests are **standalone executable scripts** with:
1. Module docstring explaining purpose
2. Helper/sub-function definitions
3. Main benchmarking function(s)
4. `if __name__ == "__main__"` entry point

**Example from `bachman_cortex/tests/benchmark_models.py`:**
```python
"""Benchmark all 4 ML models on sample frames.

Measures per-frame latency (p50/p95/p99) and total throughput for each model.
"""

def benchmark_scrfd(frames: list[np.ndarray]) -> dict:
    """Benchmark SCRFD face detector."""
    print("\n" + "=" * 60)
    print("SCRFD-2.5GF Face Detector")
    print("=" * 60)
    
    detector = SCRFDDetector(...)
    
    # Warmup
    print("Warmup (3 frames)...")
    for f in frames[:3]:
        detector.detect(f)
    
    # Benchmark
    print(f"Benchmarking on {len(frames)} frames...")
    result = detector.benchmark(frames)
    
    print(f"  p50: {result['p50_ms']:.1f}ms | p95: {result['p95_ms']:.1f}ms")
    return result

def main():
    # Extract test video
    frames, meta = extract_frames(video_path, fps=1.0, max_frames=30)
    
    # Run benchmarks on extracted frames
    results = {}
    results["scrfd"] = benchmark_scrfd(frames)
    results["yolo11m"] = benchmark_yolo(frames)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

**Patterns:**
- **Setup**: Load models once per function
- **Warmup**: Run 2-3 frames to warm GPU/CPU cache before measuring
- **Benchmark loop**: Time detector.detect() calls, collect latencies
- **Summary**: Print p50/p95/p99 percentiles and total detections
- **Output format**: Human-readable table with `print()` statements

## Mocking

**Framework:** None used in tests

**Patterns:**
- Tests use **real models and real video frames** (not mocks)
- Synthetic test data: `generate_test_video.py` creates 30-second test MP4 with simulated hands/objects
- Test frames extracted from real or synthetic video: `extract_frames(video_path, fps=1.0, max_frames=30)`

**What to Test:**
- End-to-end model inference on representative frames
- Latency measurements across frame count variations
- Detection count statistics (e.g., "total_faces": 42)
- Edge cases in motion analysis (frozen segments, jittery motion)

**What NOT to Mock:**
- ML model inference (real models tested)
- Frame extraction (real video decode tested)
- Optical flow computation (real camera stability measured)

## Fixtures and Factories

**Test Data:**
- **Synthetic video generation**: `generate_test_video.py` creates test MP4 on first run
- **Extracted frames**: Benchmarks extract frames at configurable fps/max_frames:
  ```python
  frames, meta = extract_frames(video_path, fps=1.0, max_frames=30)
  print(f"Extracted {meta['frames_extracted']} frames in {meta['extraction_time_s']}s")
  ```
- **Hardcoded paths**: Tests expect video at `bachman_cortex/sample_data/test_30s.mp4`

**Location:**
- Test data generated in `bachman_cortex/sample_data/` (not committed; created on demand)
- Video generation handled by `generate_test_video()` function
- Model weights downloaded on first run via `download_models.py`

**Example from `benchmark_models.py`:**
```python
def main():
    video_path = os.path.join(ROOT, "bachman_cortex/sample_data/test_30s.mp4")
    print(f"Extracting frames from {video_path}...")
    frames, meta = extract_frames(video_path, fps=1.0, max_frames=30)
    print(f"Extracted {meta['frames_extracted']} frames in {meta['extraction_time_s']}s")
```

## Coverage

**Requirements:** No coverage tool enforced

**Measurement:**
- No coverage reports generated
- Tests are **integration-level benchmarks**, not unit tests with coverage targets
- Manual verification of results via printed output

**View Coverage:**
- Not applicable — tests are benchmark scripts without coverage instrumentation

## Test Types

**Unit Tests:**
- **Not used** — No isolated unit test suite
- Functions tested implicitly through pipeline execution in `run_batch.py`

**Integration Tests:**
- **Benchmarks serve as integration tests**:
  - `benchmark_models.py`: Tests frame extraction + all 4 ML detectors
  - `benchmark_phase_correlation.py`: Tests motion analysis algorithms end-to-end
- Test all ML components on synthetic frames representing edge cases

**E2E Tests:**
- **End-to-end validation via `validate.sh` + `run_batch.py`**:
  - Full pipeline execution on sample video
  - Phase 0–3 executed sequentially
  - Results written to JSON and HTML reports
- Execute via: `./validate.sh /path/to/test/video.mp4`

**Benchmark Tests (Integration-Level):**
- Measure latency percentiles (p50, p95, p99) for each model
- Count detections across test frames
- Compare algorithm approaches (LK optical flow vs phase correlation)
- Report in human-readable printed tables

## Common Patterns

**Async Testing:**
- No async/await used in tests
- GPU acceleration (CUDA) tested implicitly in detector classes
- Benchmarks single-threaded; pipeline parallelism tested in `run_batch.py` with `multiprocessing.Pool`

**Error Testing:**
- Explicit exception handling tested in detector initialization:
  ```python
  if _CUDA_LK:
      try:
          solver = _get_cuda_lk(win_size, max_level)
      except Exception:
          # Fall through to CPU
          pass
  ```
- Video validation errors tested in `run_batch.py` with try/except around `process_video()`

**Timing/Latency Testing:**
```python
def benchmark(self, frames: list[np.ndarray]) -> dict:
    """Benchmark inference speed on a list of frames."""
    times = []
    for frame in frames:
        t0 = time.perf_counter()
        self.detect(frame)
        times.append(time.perf_counter() - t0)
    times_ms = [t * 1000 for t in times]
    return {
        "model": "SCRFD (buffalo_sc)",
        "frames": len(frames),
        "p50_ms": round(np.percentile(times_ms, 50), 2),
        "p95_ms": round(np.percentile(times_ms, 95), 2),
        "p99_ms": round(np.percentile(times_ms, 99), 2),
        "mean_ms": round(np.mean(times_ms), 2),
        "total_s": round(sum(times), 3),
    }
```

## Validation Script

**Location:** `validate.sh` at project root

**Pattern:**
- Bash script orchestrating setup and pipeline execution
- Multi-step validation:
  1. Check system dependencies (Python, FFmpeg, git)
  2. Create/verify virtual environment
  3. Install PyTorch (GPU-aware)
  4. Install ONNX Runtime, detectron2, hl-video-validation package
  5. Download model weights
  6. Run pipeline: `hl-validate <video_paths>`

**Usage:**
```bash
./validate.sh /path/to/video.mp4          # Single video
./validate.sh /path/to/videos/            # Directory of videos
./validate.sh --setup-only                # Setup without running pipeline
FORCE_CPU=1 ./validate.sh /path/to/vid    # CPU-only mode
```

**Key Features:**
- Idempotent: safe to run multiple times (skips already-installed steps)
- GPU-aware: detects NVIDIA GPU and installs appropriate PyTorch version
- Fallback to CPU if no GPU or `FORCE_CPU=1` set
- Per-worker model initialization in `run_batch.py` via `_init_worker()`

## Running Tests

**Benchmark Models:**
```bash
python -m bachman_cortex.tests.benchmark_models
```
Output:
```
SCRFD-2.5GF Face Detector
==============================================
Warmup (3 frames)...
Benchmarking on 30 frames...
  p50: 45.3ms | p95: 52.1ms | mean: 46.2ms
  Total faces detected: 42
```

**Phase-Correlation Benchmark:**
```bash
python -m bachman_cortex.tests.benchmark_phase_correlation
```
Compares LK optical flow + SSIM motion analysis vs phase-correlation approach.

**Generate Test Video:**
```bash
python -m bachman_cortex.tests.generate_test_video
```
Creates synthetic 30s 1080p 30fps test video at `bachman_cortex/sample_data/test_30s.mp4`.

## Test Data Location

```
bachman_cortex/
├── sample_data/
│   └── test_30s.mp4              # Synthetic test video (generated on demand)
├── models/
│   └── weights/
│       ├── insightface/          # SCRFD model weights (downloaded on setup)
│       ├── hands23_detector/     # Hands23 model (downloaded on setup)
│       └── ...
```

Model weights are downloaded via `bachman_cortex/models/download_models.py` during setup (called from `validate.sh`).

---

*Testing analysis: 2026-04-15*
