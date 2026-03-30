# Egocentric Video ML Checks Pipeline

A pipeline for validating egocentric (first-person POV) video quality using 4 ML models across 7 acceptance criteria checks. Built for datasets used to train autonomous humanoid robots.

## The 7 Checks

| # | Check | Criterion | Model |
|---|---|---|---|
| 1 | Face Presence | No face with confidence >= 0.8 in any frame | SCRFD-2.5GF |
| 2 | Participants | 0 other persons in >= 95% of frames | YOLO11m + SCRFD |
| 3 | Hand Visibility | 2+ hands detected in >= 90% of frames | Hands23 |
| 4 | Hand-Object Interaction | Interaction in >= 70% of frames | Hands23 (contact state) |
| 5 | Privacy Safety | 0 sensitive objects in all frames | YOLO11m + Grounding DINO |
| 6 | View Obstruction | <= 10% frames obstructed | OpenCV heuristic |
| 7 | POV-Hand Angle | Hands within 40 degrees of center in >= 80% of frames | Geometric (on Hands23 output) |

## Prerequisites

- Python 3.10+
- FFmpeg (for video rotation and frame extraction)
- ~1.5 GB disk space for model weights

## Setup

```bash
git clone <repo-url> && cd <repo-name>

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CPU — for GPU, see https://pytorch.org/get-started)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r ml_checks/requirements.txt

# Install detectron2 (required for Hands23 hand detector)
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```

### Download models

```bash
# Download all models: SCRFD, YOLO11m, Grounding DINO, Hands23
# Hands23 repo is cloned, weights downloaded, and .cuda() calls patched automatically
python ml_checks/models/download_models.py --all
```

## Usage

### Single video

```bash
python -m ml_checks.run_batch /path/to/video.mp4
```

### Multiple videos

```bash
python -m ml_checks.run_batch video1.mp4 video2.mp4 video3.mp4
```

### Directory of videos

```bash
python -m ml_checks.run_batch /path/to/videos/
```

### With options

```bash
# Auto-rotate portrait videos to landscape before processing
python -m ml_checks.run_batch /path/to/videos/ --auto-rotate

# Custom output directory
python -m ml_checks.run_batch /path/to/videos/ --output my_results/

# Faster: skip Grounding DINO (no fine-grained privacy detection)
python -m ml_checks.run_batch /path/to/videos/ --no-gdino

# Quick test: limit to 10 frames per video
python -m ml_checks.run_batch /path/to/videos/ --max-frames 10

# Higher accuracy: sample at 2 FPS instead of 1
python -m ml_checks.run_batch /path/to/videos/ --fps 2

# Combine options
python -m ml_checks.run_batch /path/to/videos/ --auto-rotate --fps 1 --output results/ --no-gdino
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--output`, `-o` | `ml_checks/results` | Output directory for reports and JSON |
| `--fps` | `1.0` | Frame sampling rate (frames per second) |
| `--max-frames` | no limit | Max frames to sample per video |
| `--no-gdino` | off | Disable Grounding DINO (faster, skips fine-grained privacy detection) |
| `--auto-rotate` | off | Auto-rotate portrait videos to landscape before processing |
| `--hand-detector-repo` | `ml_checks/models/weights/hands23_detector` | Path to Hands23 repo |
| `--scrfd-root` | `ml_checks/models/weights/insightface` | Path to InsightFace models |
| `--gdino-cache` | `ml_checks/models/weights/grounding_dino` | Path to Grounding DINO cache |

## Output

```
results/
├── batch_report.md          # Markdown summary of all videos
├── batch_results.json       # Full JSON with all results
├── video1.json              # Per-video detailed results
├── video2.json
└── ...
```

### batch_report.md

Human-readable summary table with pass/fail status for every check on every video, plus details on failing checks.

### Per-video JSON

```json
{
  "filename": "video.mp4",
  "video_meta": { "width": 1920, "height": 1080, "fps": 29.57, "duration_s": 25.8 },
  "processing_time_s": 41.1,
  "results": {
    "face_presence": { "status": "pass", "metric_value": 1.0, "confidence": 0.71, "details": {...} },
    "participants": { "status": "pass", "metric_value": 0.96, "confidence": 0.77, "details": {...} },
    ...
  }
}
```

## Models

| Model | What it does | Size | Speed (CPU) | Speed (GPU est.) |
|---|---|---|---|---|
| [SCRFD-2.5GF](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) | Face detection | ~14 MB | ~10 ms/frame | ~3 ms/frame |
| [YOLO11m](https://docs.ultralytics.com/models/yolo11/) | Person + object detection | ~40 MB | ~88 ms/frame | ~5 ms/frame |
| [Hands23](https://github.com/EvaCheng-cty/hands23_detector) | Hand detection + contact state + grasp | ~446 MB | ~1,400 ms/frame | ~100-200 ms/frame |
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | Zero-shot privacy object detection | ~700 MB | ~2,750 ms/frame | ~300 ms/frame |

### Why these models

- **SCRFD** over MediaPipe: 93.8% AP on WIDER Face hard set vs MediaPipe's webcam-optimized detection. Better for bystander faces at egocentric angles.
- **YOLO11m** over nano: 51.5 vs 39.5 mAP. Person detection is safety-critical.
- **Hands23** (NeurIPS 2023) over 100DOH/MediaPipe: Successor to 100DOH by the same research group. Trained on 250K images (2.5x more data) including EPIC-KITCHENS and VISOR egocentric datasets. Outputs hand bboxes, left/right side, contact state, grasp type, and associated object bboxes. 3.5x faster than 100DOH on CPU. Runs on standard Detectron2 (no custom C++ compilation).
- **Grounding DINO** for privacy: Zero-shot detection of "credit card", "ID card", "paper document" without fine-tuning. Text prompts can be updated without retraining.

A legacy 100DOH backend is preserved in `hand_detector_100doh.py` and can be swapped back in `pipeline.py`.

See [CONTEXT.md](ml_checks/CONTEXT.md) for full model selection rationale, alternatives considered, and benchmark data.

## Performance

For a 5-minute video sampled at 1 FPS (300 frames):

| | CPU (Apple M-series) | GPU (A10G, projected) |
|---|---|---|
| Total time | ~8 min | ~40-65s |
| Bottleneck | Hands23 (~1.4s/frame) | Hands23 (~100-200ms/frame) |
| Cost at 5K videos/day | ~$1,500/mo | ~$870/mo |

Use `--max-frames 10` for quick testing.

## Switching hand detection backends

The pipeline ships with two hand detection backends. **Hands23** is active by default.

### Hands23 (default, NeurIPS 2023)

- Faster R-CNN X-101-FPN on Detectron2
- 250K training images including EPIC-KITCHENS and VISOR egocentric data
- Outputs: hand bbox, left/right, contact state, grasp type, object bbox
- ~1.4s/frame CPU, ~100-200ms GPU
- No custom C++ compilation needed

### 100DOH (legacy, CVPR 2020)

- Faster R-CNN ResNet-101, custom codebase
- 100K+ego training images
- Outputs: hand bbox, left/right (unreliable), contact state, offset vectors
- ~5.1s/frame CPU, ~100-150ms GPU
- Requires C++ extension compilation and `DYLD_LIBRARY_PATH` on macOS

### How to switch to 100DOH

1. Download and compile 100DOH:
   ```bash
   python ml_checks/models/download_models.py --100doh
   ```

2. Edit `ml_checks/pipeline.py` — comment out Hands23, uncomment 100DOH:
   ```python
   # Comment out Hands23:
   # from ml_checks.models.hand_detector import HandObjectDetectorHands23
   # self.hand_detector = HandObjectDetectorHands23(
   #     repo_dir=self.config.hand_detector_repo,
   # )

   # Uncomment 100DOH:
   from ml_checks.models.hand_detector_100doh import HandObjectDetector100DOH
   self.hand_detector = HandObjectDetector100DOH(
       repo_dir="ml_checks/models/weights/hand_object_detector",
   )
   ```

3. On macOS, set the library path before running:
   ```bash
   export DYLD_LIBRARY_PATH=$(python -c "import torch; print(torch.__file__.replace('__init__.py','lib'))")
   ```

### How to run both simultaneously (A/B comparison)

To compare both backends on the same videos, run the pipeline twice with different output directories:

```bash
# Run 1: Hands23 (default)
python -m ml_checks.run_batch /path/to/videos/ --output results_hands23/

# Run 2: 100DOH — edit pipeline.py to use 100DOH (see above), then:
python -m ml_checks.run_batch /path/to/videos/ --output results_100doh/

# Compare the two batch_report.md files side by side
diff results_hands23/batch_report.md results_100doh/batch_report.md
```

Or for a quick programmatic comparison without editing `pipeline.py`:

```python
from ml_checks.models.hand_detector import HandObjectDetectorHands23
from ml_checks.models.hand_detector_100doh import HandObjectDetector100DOH
from ml_checks.utils.frame_extractor import extract_frames

frames, _ = extract_frames("video.mp4", fps=1.0, max_frames=5)

hands23 = HandObjectDetectorHands23()
doh100 = HandObjectDetector100DOH()

for i, frame in enumerate(frames):
    h23_hands, _ = hands23.detect(frame)
    doh_hands, _ = doh100.detect(frame)
    print(f"Frame {i+1}: Hands23={len(h23_hands)} hands, 100DOH={len(doh_hands)} hands")
```

Both backends produce the same `HandDetection` and `ObjectDetection` types with the same `ContactState`, `HandSide` enums, so all 7 check functions work identically with either backend.

## Portrait videos

The Hands23 hand detector was trained on landscape images. Portrait videos produce worse hand detection results. Use `--auto-rotate` to automatically detect and rotate portrait videos before processing.

## Project structure

```
ml_checks/
├── README.md
├── CONTEXT.md                      # Full research context and model rationale
├── requirements.txt
├── pipeline.py                     # Unified inference pipeline
├── run_batch.py                    # Batch processing script
├── checks/
│   ├── check_results.py            # CheckResult dataclass
│   ├── face_presence.py            # Check 1
│   ├── participants.py             # Check 2
│   ├── hand_visibility.py          # Check 3
│   ├── hand_object_interaction.py  # Check 4
│   ├── privacy_safety.py           # Check 5
│   ├── view_obstruction.py         # Check 6
│   └── pov_hand_angle.py           # Check 7
├── models/
│   ├── download_models.py          # Model weight downloader
│   ├── scrfd_detector.py           # SCRFD wrapper
│   ├── yolo_detector.py            # YOLO11m wrapper
│   ├── hand_detector.py            # Hands23 wrapper (active)
│   ├── hand_detector_100doh.py     # 100DOH wrapper (legacy, commented out)
│   └── grounding_dino_detector.py  # Grounding DINO wrapper
├── utils/
│   └── frame_extractor.py          # Video frame sampling
└── tests/
    ├── generate_test_video.py      # Synthetic test video generator
    └── benchmark_models.py         # Model benchmarking script
```

## License

The pipeline code is provided as-is. Model weights are subject to their respective licenses:
- SCRFD/InsightFace: MIT
- YOLO11m/Ultralytics: AGPL-3.0
- Hands23: MIT
- Grounding DINO: Apache-2.0
