# Bachman — Egocentric Video Scoring Engine

Scores egocentric (first-person POV) videos against a fixed set of
acceptance criteria and produces a per-video report plus a batch
rollup. Reads MP4s, writes Markdown + JSON + Parquet. No clips are
extracted — the engine reports observations, not decisions.

The full design spec lives in [`SCORING_ENGINE_PLAN.md`](SCORING_ENGINE_PLAN.md). Check specifications
live in [`checks.md`](checks.md).

---

## Quick start

```bash
git clone <repo-url> && cd hl-bachman
./validate.sh /path/to/videos/
```

First run bootstraps a virtualenv, installs PyTorch / ONNX Runtime /
detectron2, and downloads ~440 MB of model weights. Subsequent runs
start in seconds.

After setup, the CLI is `hl-score`:

```bash
source .venv/bin/activate
hl-score path/to/video.mp4
hl-score path/to/dir/
hl-score path/to/dir/ --config ./hl-score.toml
hl-score --dump-default-config > hl-score.toml
```

---

## Requirements

- **Python 3.11+** (3.12 / 3.13 also supported).
- **FFmpeg** (`ffprobe` for metadata).
- **~2 GB disk** for model weights + venv.
- Linux or macOS (Apple Silicon). On Windows, use WSL2 — detectron2
  has no native Windows support.
- GPU optional — auto-detected. Force CPU with
  `FORCE_CPU=1 ./validate.sh ...`.

---

## How it works

The engine runs three **gated** stages on each video:

### Stage 1 — Metadata

Cheap ffprobe reads. Any single failure skips stages 2 and 3 and no
decode is performed.

| Check       | Rule                                                       |
| ----------- | ---------------------------------------------------------- |
| format      | Container is MP4.                                           |
| encoding    | H.264 or HEVC.                                              |
| resolution  | Displayed dims ≥ 1920×1080 (after rotation).                |
| frame_rate  | ≥ 28 FPS.                                                   |
| duration    | ≥ 59 s.                                                      |
| orientation | Rotation ∈ {0, 90, 270} AND landscape after rotation.        |

### Stage 2 — Technical

Runs against every video that passes metadata, in a single streaming
decode. Any one failure marks the video technical-failed; quality
metrics are then rendered as `SKIPPED` in MD/JSON (but the raw
per-frame columns still land in parquet).

| Check      | Rule                                                                     |
| ---------- | ------------------------------------------------------------------------ |
| luminance  | ≥ 80% good frames (not dead black / too dark / blown out / flicker).      |
| stability  | Whole-video mean jitter score ≤ 0.181 (high-pass-filtered LK).             |
| frozen     | No run > 30 consecutive native-frame-equivalents of near-zero motion.      |
| pixelation | ≥ 80% frames with blockiness ratio ≤ 1.5.                                  |

### Stage 3 — Quality metrics

Six per-frame metrics at 2 FPS on 720p frames. Reported as
`percent_frames` (from the un-merged per-frame array) plus merged
segments. Not pass/fail at the video level — these are observations.

| Metric                  | Per-frame condition                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------------------- |
| both_hands_visibility   | Hands23 detects L AND R, each conf ≥ 0.7.                                                                |
| single_hand_visibility  | ≥1 hand with Hands23 conf ≥ 0.7.                                                                         |
| hand_obj_interaction    | ≥1 detected hand with contact state ∈ {P, F}.                                                            |
| hand_angle              | All detected hands within 40° of frame centre.                                                           |
| participants            | ≥1 other-person signal (YOLO ≥0.6, SCRFD ≥0.6, Hands23 extra-hand ≥0.7). Wearer-filtered on YOLO + SCRFD. |
| obstructed              | ≥2 of 4 heuristic signals triggered on the central 80% crop.                                             |

---

## Output layout

Each run allocates a fresh `run_NNN/` directory under `--out-dir`:

```
results/run_NNN/
├── {video_name}/
│   ├── report.md
│   ├── {video_name}.json
│   └── {video_name}.parquet     # omitted if metadata failed
├── batch_report.md
├── batch_results.json
└── batch_results.csv
```

The parquet file carries one row per native frame, dense, with
per-frame jitter / frozen flag / luminance class / pixelation ratio
/ all 6 quality metric columns. Null-filled where a cadence didn't
tick that frame. See `SCORING_ENGINE_PLAN.md` §4 for the full schema.

---

## CLI flags

| Flag                     | Default       | Meaning                                                  |
| ------------------------ | ------------- | -------------------------------------------------------- |
| *(positional)* `INPUT`   | —             | Single MP4 file or directory to walk recursively.         |
| `--out-dir`              | `results/`    | Output root; `run_NNN` is created under this.             |
| `--config`               | (none)        | TOML config with cadences + thresholds.                   |
| `--workers`              | auto          | Reserved for future multi-process mode.                   |
| `--hand-detector-repo`   | (built-in)    | Override Hands23 weights path.                            |
| `--scrfd-root`           | (built-in)    | Override SCRFD weights path.                              |
| `--yolo-model`           | `yolo11m.pt`  | Override YOLO weights.                                    |
| `--dump-default-config`  | —             | Write the default TOML template to stdout and exit.       |
| `--verbose` / `-v`       | off           | Log per-video progress.                                   |

Only top-level knobs are on the CLI; fine-grained thresholds live in
the TOML config.

---

## Models

| Model         | Purpose                           | Size    |
| ------------- | --------------------------------- | ------- |
| SCRFD-2.5GF   | Face detection                     | ~14 MB  |
| YOLO11        | Person detection                   | ~20 MB  |
| Hands23       | Hand detection + contact state     | ~400 MB |

Total ~440 MB downloaded on first run.

---

## Custom CUDA OpenCV (optional)

PyPI's `opencv-python` doesn't ship with CUDA. To enable
`cv2.cuda.SparsePyrLKOpticalFlow` (GPU optical flow) and
`cv2.cudacodec.VideoReader` (NVDEC hardware decode), build OpenCV
from source:

```bash
# Download NVIDIA Video Codec SDK 12.x headers (free dev-portal login):
#   https://developer.nvidia.com/video-codec-sdk-archive
NVCODEC_SDK="$PWD/Video_Codec_SDK_12.2.72" ./scripts/install_opencv_cuda.sh
```

See [`scripts/install_opencv_cuda.sh`](scripts/install_opencv_cuda.sh)
for the details; the engine auto-detects NVDEC availability and
falls back to `cv2.VideoCapture` when the build doesn't have it.

The pipeline ships a numpy `cv2.dnn.blobFromImage` shim
(`bachman_cortex/_cv2_dnn_shim.py`) so insightface still works with
the dnn-less custom build.

---

## Project layout

```
hl-bachman/
├── validate.sh                  # One-command setup + run.
├── pyproject.toml
├── SCORING_ENGINE_PLAN.md       # Locked design spec.
├── checks.md                    # Check specifications.
├── README.md                    # This file.
├── scripts/
│   └── install_opencv_cuda.sh   # Builds custom OpenCV + NVDEC.
└── bachman_cortex/
    ├── __init__.py              # Installs cv2.dnn shim.
    ├── _cv2_dnn_shim.py         # Pure-numpy blobFromImage.
    ├── cli.py                   # hl-score entrypoint.
    ├── batch.py                 # Directory walk + batch driver.
    ├── scoring_engine.py        # Single-video orchestrator.
    ├── config.py                # TOML config schema + loader.
    ├── data_types.py            # Report dataclasses.
    ├── reporting.py             # MD / JSON / CSV writers.
    ├── segmentation.py          # group_runs + merge_short_runs.
    ├── per_frame_store.py       # In-memory row buffer + parquet.
    ├── checks/
    │   ├── check_results.py
    │   ├── video_metadata.py    # Six metadata checks.
    │   ├── motion_analysis.py   # MotionAnalyzer (stability + frozen).
    │   ├── luminance.py         # LuminanceAccumulator.
    │   ├── pixelation.py        # PixelationAccumulator.
    │   ├── view_obstruction.py  # ObstructionAccumulator.
    │   ├── hand_visibility.py   # HandsAccumulator (both/single/angle/HOI).
    │   └── participants.py      # ParticipantsAccumulator + wearer filter.
    ├── models/
    │   ├── download_models.py   # Weight downloader.
    │   ├── hand_detector.py     # Hands23 via Detectron2.
    │   ├── scrfd_detector.py    # SCRFD via InsightFace.
    │   └── yolo_detector.py     # YOLO11 via Ultralytics.
    └── utils/
        ├── frame_extractor.py   # iter_native_frames (NVDEC/cv2 generator).
        └── video_metadata.py    # ffprobe wrapper.
```

---

## Testing

```bash
source .venv/bin/activate
pytest -q
```

Unit tests cover: TOML config roundtrip, per-frame parquet store,
motion analyser finalisation, each quality / technical accumulator,
the §1 segment-merging absorption rules, reporting MD / JSON / CSV
formatting, and batch input discovery.

End-to-end validation (decode + model inference + report emission)
is exercised manually via `hl-score` on real clips; the automated
tests are deterministic and don't require GPUs or model weights.
