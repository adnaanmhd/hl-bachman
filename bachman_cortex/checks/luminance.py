"""Luminance & flicker quality check.

Two signals combined into a single pass/fail verdict:
1. Luminance — per-frame dark/blown-out rejection
2. Brightness flicker — rolling stddev detection

Video-level: pass if good frames >= min_good_ratio of total.
"""

import cv2
import numpy as np
from dataclasses import dataclass

from bachman_cortex.checks.check_results import CheckResult


@dataclass
class LuminanceConfig:
    """Thresholds for luminance and flicker checks."""
    # Luminance zones (0-255 grayscale mean)
    lum_dead_black: float = 15.0
    lum_too_dark: float = 45.0
    lum_blown_out: float = 230.0

    # Brightness stability — flicker
    flicker_window: int = 10
    flicker_std_threshold: float = 30.0

    # Video-level
    min_good_ratio: float = 0.80


@dataclass
class FrameMetrics:
    """Per-frame luminance metrics."""
    frame_idx: int
    mean_luminance: float
    label: str          # accept, reject
    reject_reason: str  # "" if accepted


def compute_frame_metrics(frame: np.ndarray, frame_idx: int) -> FrameMetrics:
    """Compute luminance for a single frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))

    return FrameMetrics(
        frame_idx=frame_idx,
        mean_luminance=mean_lum,
        label="",
        reject_reason="",
    )


def classify_luminance(m: FrameMetrics, cfg: LuminanceConfig) -> FrameMetrics:
    """Classify a frame by luminance."""
    lum = m.mean_luminance

    if lum < cfg.lum_dead_black:
        m.label, m.reject_reason = "reject", "dead_black"
    elif lum < cfg.lum_too_dark:
        m.label, m.reject_reason = "reject", "too_dark"
    elif lum > cfg.lum_blown_out:
        m.label, m.reject_reason = "reject", "blown_out"
    else:
        m.label = "accept"

    return m


def detect_flicker(
    luminances: list[float],
    cfg: LuminanceConfig,
) -> list[bool]:
    """Detect visible brightness flicker via rolling stddev.

    A window of flicker_window frames slides across the luminance series.
    If the stddev within the window exceeds flicker_std_threshold, all
    frames in that window are marked True.
    """
    n = len(luminances)
    if n < cfg.flicker_window:
        return [False] * n

    flicker = [False] * n
    lum_arr = np.array(luminances)

    for start in range(n - cfg.flicker_window + 1):
        window = lum_arr[start:start + cfg.flicker_window]
        if float(np.std(window)) > cfg.flicker_std_threshold:
            for k in range(start, start + cfg.flicker_window):
                flicker[k] = True

    return flicker


def check_luminance(
    frames: list[np.ndarray],
    config: LuminanceConfig | None = None,
    timestamps: list[float] | None = None,
) -> CheckResult:
    """Run luminance and flicker checks on sampled frames.

    Pass if good frames (no luminance reject, no flicker) >= min_good_ratio
    of total frames.
    """
    cfg = config or LuminanceConfig()

    if not frames:
        return CheckResult(status="fail", metric_value=0.0, confidence=1.0,
                           details={"error": "no frames"})

    # Per-frame metrics + luminance classification
    all_metrics = []
    for i, frame in enumerate(frames):
        m = compute_frame_metrics(frame, i)
        m = classify_luminance(m, cfg)
        all_metrics.append(m)

    luminances = [m.mean_luminance for m in all_metrics]

    # Flicker detection
    flicker = detect_flicker(luminances, cfg)

    # Merge reject reasons per frame
    reject_reasons: dict[str, int] = {}
    rejected = [False] * len(frames)

    for i, m in enumerate(all_metrics):
        # Luminance rejects
        if m.label == "reject":
            rejected[i] = True
            reject_reasons[m.reject_reason] = reject_reasons.get(m.reject_reason, 0) + 1
            continue

        # Flicker
        if flicker[i]:
            rejected[i] = True
            reject_reasons["brightness_flicker"] = reject_reasons.get("brightness_flicker", 0) + 1
            m.label, m.reject_reason = "reject", "brightness_flicker"
            continue

    total = len(frames)
    reject_count = sum(rejected)
    accept_count = total - reject_count
    good_ratio = accept_count / total

    status = "pass" if good_ratio >= cfg.min_good_ratio else "fail"

    # Failure types present
    failure_types = list(reject_reasons.keys())

    # Flicker segments for diagnostics
    flicker_segments = _find_runs(flicker, timestamps)

    return CheckResult(
        status=status,
        metric_value=round(good_ratio, 4),
        confidence=1.0,
        details={
            "accept_frames": accept_count,
            "reject_frames": reject_count,
            "total_frames": total,
            "good_ratio": round(good_ratio, 4),
            "min_good_ratio": cfg.min_good_ratio,
            "reject_reasons": reject_reasons,
            "failure_types": failure_types,
            "flicker_segments": flicker_segments,
        },
    )


def _find_runs(
    mask: list[bool],
    timestamps: list[float] | None,
) -> list[dict]:
    """Collapse a boolean mask into contiguous run dicts."""
    runs: list[dict] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        run: dict = {"start_frame": i, "end_frame": j, "length": j - i}
        if timestamps is not None:
            run["start_sec"] = timestamps[i] if i < len(timestamps) else None
            run["end_sec"] = timestamps[j - 1] if j - 1 < len(timestamps) else None
        runs.append(run)
        i = j
    return runs
