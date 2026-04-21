"""Batch driver for the scoring engine.

Walks a directory (recursively) OR takes a single file, scores each
video in turn, and writes per-video + batch reports into
`{out_root}/run_NNN`. Per-video failure (decode error, corrupt,
audio-only) does not abort the batch — the error is captured in
`BatchScoreReport.errors`.

Why serial by default: each Hands23 inference pins a GPU and the
models are heavy enough that within-process threading mostly
contends. Parallelism for a future revision would be multi-process
with each worker getting its own engine; the `--workers` knob is
reserved but not yet wired so we don't regress correctness while
the single-pass path is still landing.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from bachman_cortex.config import Config
from bachman_cortex.data_types import (
    BatchScoreReport,
    ProcessingErrorReport,
    VideoScoreReport,
)
from bachman_cortex.reporting import (
    aggregate_batch_stats,
    allocate_run_dir,
    write_batch_report,
    write_video_report,
)
from bachman_cortex.scoring_engine import ScoringEngine


log = logging.getLogger("bachman_cortex.batch")


# ── Input discovery ───────────────────────────────────────────────────────

def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def iter_input_videos(root: str | Path) -> Iterator[Path]:
    """Yield MP4 paths under `root`.

    - Accepts `.mp4` case-insensitive.
    - Skips hidden files / directories (dot-prefix).
    - Follows symlinks but guards against cycles via an inode-visited set.
    - Accepts a single file or a directory.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"input path does not exist: {root}")

    if root_path.is_file():
        if root_path.suffix.lower() == ".mp4":
            yield root_path
        return

    visited: set[tuple[int, int]] = set()

    def _walk(directory: Path) -> Iterator[Path]:
        try:
            st = directory.stat()
        except OSError:
            return
        key = (st.st_dev, st.st_ino)
        if key in visited:
            return
        visited.add(key)

        try:
            entries = sorted(directory.iterdir())
        except (PermissionError, OSError):
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                yield from _walk(entry)
            elif entry.is_file() and entry.suffix.lower() == ".mp4":
                yield entry

    yield from _walk(root_path)


# ── Batch driver ──────────────────────────────────────────────────────────

@dataclass
class BatchOptions:
    out_root: str | Path = "results"
    config: Config | None = None
    workers: int | None = None           # reserved — see module docstring
    hand_detector_repo: str | None = None
    scrfd_root: str | None = None
    yolo_model: str | None = None


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def score_batch(
    inputs: Iterable[str | Path],
    options: BatchOptions | None = None,
) -> tuple[BatchScoreReport, Path]:
    """Score each discovered video. Returns (batch_report, run_dir)."""
    opts = options or BatchOptions()
    cfg = opts.config or Config()

    # Expand inputs → deduplicated, input-order-preserving list of videos.
    videos: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        for v in iter_input_videos(item):
            abs_v = v.resolve()
            if abs_v in seen:
                continue
            seen.add(abs_v)
            videos.append(v)

    run_dir = allocate_run_dir(opts.out_root)
    log.info("batch run dir: %s", run_dir)
    log.info("videos: %d", len(videos))

    engine = ScoringEngine(
        config=cfg,
        hand_detector_repo=opts.hand_detector_repo,
        scrfd_root=opts.scrfd_root,
        yolo_model=opts.yolo_model,
    )

    reports: list[VideoScoreReport] = []
    errors: list[ProcessingErrorReport] = []
    t_batch = time.perf_counter()

    for video in videos:
        try:
            report, store, imu_samples = engine.score_video(video)
        except Exception as exc:
            reason = _classify_error(exc)
            log.exception("scoring failed: %s (%s)", video.name, reason)
            errors.append(ProcessingErrorReport(
                video_path=str(video),
                video_name=video.name,
                error_reason=reason,
            ))
            continue

        try:
            write_video_report(
                report, run_dir,
                per_frame_store=store,
                imu_samples=imu_samples,
            )
        except Exception:
            log.exception("failed to write per-video report for %s", video.name)
            errors.append(ProcessingErrorReport(
                video_path=str(video),
                video_name=video.name,
                error_reason="report_write_failed",
            ))
            continue

        reports.append(report)

    meta_stats, tech_stats, qual_stats = aggregate_batch_stats(reports)
    batch = BatchScoreReport(
        generated_at=_iso_utc_now(),
        video_count=len(reports),
        total_duration_s=round(sum(r.duration_s for r in reports), 2),
        total_wall_time_s=round(time.perf_counter() - t_batch, 3),
        metadata_check_stats=meta_stats,
        technical_check_stats=tech_stats,
        quality_metric_stats=qual_stats,
        videos=reports,
        errors=errors,
    )
    write_batch_report(batch, run_dir)
    return batch, run_dir


def _classify_error(exc: BaseException) -> str:
    """Map an exception to a short stable `error_reason` string."""
    msg = str(exc).lower()
    if "no video stream" in msg or "audio" in msg:
        return "audio_only"
    if "cannot open" in msg or "decode" in msg:
        return "decode_failed"
    if "ffprobe" in msg:
        return "metadata_probe_failed"
    if "corrupt" in msg or "invalid" in msg:
        return "corrupt"
    return type(exc).__name__


# ── CPU + GPU auto-detect (used by CLI) ───────────────────────────────────

def auto_worker_count() -> int:
    """Plan §1: workers = max(1, min(cpu_count // 2, gpu_count * 2)).

    The knob is reserved — the batch driver is currently serial — but
    the helper lives here so the CLI can surface a consistent default.
    """
    cpu = os.cpu_count() or 1
    try:
        import torch
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        gpus = 0
    return max(1, min(cpu // 2, gpus * 2) if gpus > 0 else max(1, cpu // 2))
