"""Motion analysis checks.

Camera stability via Farneback dense optical flow on sampled frames.
Frozen segment detection via SSIM at native FPS (streaming, memory-efficient).
"""

import cv2
import numpy as np
from pathlib import Path

from ml_checks.checks.check_results import CheckResult


def _compute_ssim_gray(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two grayscale images using OpenCV.

    Lightweight implementation -- no scikit-image dependency.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def check_camera_stability(
    frames: list[np.ndarray],
    threshold_px: float = 15.0,
    pass_rate: float = 0.80,
) -> CheckResult:
    """Check camera stability using Farneback dense optical flow.

    Computes mean optical flow magnitude between consecutive sampled frame pairs.
    Pass if >= pass_rate fraction of pairs have mean flow <= threshold_px.

    Args:
        frames: Sampled BGR frames (e.g., 1 FPS).
        threshold_px: Max mean flow magnitude for a stable pair.
        pass_rate: Fraction of pairs that must be stable.
    """
    if len(frames) < 2:
        return CheckResult(status="pass", metric_value=1.0, confidence=1.0,
                           details={"error": "fewer than 2 frames, skipping"})

    stable_pairs = 0
    total_pairs = len(frames) - 1
    flow_magnitudes = []

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_mag = float(np.mean(magnitude))
        flow_magnitudes.append(mean_mag)

        if mean_mag <= threshold_px:
            stable_pairs += 1

        prev_gray = curr_gray

    stable_ratio = stable_pairs / total_pairs
    passes = stable_ratio >= pass_rate

    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=round(stable_ratio, 4),
        confidence=1.0,
        details={
            "stable_pairs": stable_pairs,
            "total_pairs": total_pairs,
            "stable_ratio": round(stable_ratio, 4),
            "threshold_px": threshold_px,
            "pass_rate": pass_rate,
            "mean_flow_magnitude": round(float(np.mean(flow_magnitudes)), 2),
            "max_flow_magnitude": round(float(np.max(flow_magnitudes)), 2),
        },
    )


def check_frozen_segments(
    video_path: str | Path,
    max_consecutive: int = 30,
    ssim_threshold: float = 0.99,
    downscale_height: int = 480,
) -> CheckResult:
    """Check for frozen segments by reading video at native FPS.

    Streams frames sequentially and computes SSIM between consecutive pairs.
    Fails if any run of consecutive frames exceeds max_consecutive with
    SSIM > ssim_threshold.

    Optimizations:
    - Downscale to 480p grayscale before SSIM.
    - Fast pre-filter: skip SSIM if mean absolute difference > 5.
    - Streams 2 frames at a time (constant memory).

    Args:
        video_path: Path to video file.
        max_consecutive: Max allowed consecutive frozen frames.
        ssim_threshold: SSIM above this = frozen.
        downscale_height: Height to downscale to before comparison.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS)

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return CheckResult(status="pass", metric_value=0.0, confidence=1.0,
                           details={"error": "could not read first frame"})

    prev_gray = _downscale_gray(prev_frame, downscale_height)

    current_run = 0
    longest_run = 0
    frozen_segments = []  # (start_frame, length)
    run_start = 0
    frames_read = 1

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        frames_read += 1

        curr_gray = _downscale_gray(curr_frame, downscale_height)

        # Fast pre-filter: if frames are clearly different, skip SSIM
        mean_abs_diff = float(np.mean(np.abs(
            curr_gray.astype(np.float32) - prev_gray.astype(np.float32)
        )))

        if mean_abs_diff <= 5.0:
            ssim = _compute_ssim_gray(prev_gray, curr_gray)
            is_frozen = ssim > ssim_threshold
        else:
            is_frozen = False

        if is_frozen:
            if current_run == 0:
                run_start = frames_read - 2
            current_run += 1
        else:
            if current_run > 0:
                if current_run > longest_run:
                    longest_run = current_run
                if current_run > max_consecutive:
                    frozen_segments.append({"start_frame": run_start, "length": current_run})
                current_run = 0

        prev_gray = curr_gray

    # Handle run at end of video
    if current_run > 0:
        if current_run > longest_run:
            longest_run = current_run
        if current_run > max_consecutive:
            frozen_segments.append({"start_frame": run_start, "length": current_run})

    cap.release()

    passes = longest_run <= max_consecutive
    metric = longest_run / max_consecutive if max_consecutive > 0 else 0.0

    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=round(min(metric, 10.0), 4),  # cap at 10x for readability
        confidence=1.0,
        details={
            "longest_frozen_run": longest_run,
            "max_consecutive_allowed": max_consecutive,
            "ssim_threshold": ssim_threshold,
            "total_frames_read": frames_read,
            "native_fps": round(native_fps, 2),
            "frozen_duration_s": round(longest_run / native_fps, 2) if native_fps > 0 else 0,
            "frozen_segments": frozen_segments[:10],
        },
    )


def _downscale_gray(frame: np.ndarray, target_height: int) -> np.ndarray:
    """Convert to grayscale and downscale to target height."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h <= target_height:
        return gray
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_AREA)
