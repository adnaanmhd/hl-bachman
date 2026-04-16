"""Pixelation detection via block boundary gradient ratio.

Measures the ratio of gradient energy at 8x8 block boundaries vs. interior
positions. Compression artifacts and upscaled low-res sources produce
artificially strong edges at block boundaries, yielding a ratio >> 1.0.

Video-level: pass if mean blockiness score <= threshold.
"""

import cv2
import numpy as np
from dataclasses import dataclass

from bachman_cortex.checks.check_results import CheckResult


@dataclass
class PixelationConfig:
    """Thresholds for pixelation detection."""
    # Block size to check (8 = H.264/HEVC sub-block)
    block_size: int = 8

    # Blockiness ratio above this = pixelated frame
    pixelation_threshold: float = 1.5

    # Video-level: fraction of frames that must be non-pixelated
    min_good_ratio: float = 0.80


def compute_blockiness(frame: np.ndarray, block_size: int = 8) -> float:
    """Compute blockiness score for a single frame.

    Measures the ratio of average gradient magnitude at block boundaries
    vs. interior positions. Returns ratio >= 0.0; values near 1.0 indicate
    no block artifacts, values >> 1.0 indicate pixelation.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # Horizontal gradients: |I(x+1, y) - I(x, y)|
    h_grad = np.abs(np.diff(gray, axis=1))  # shape (h, w-1)

    # Vertical gradients: |I(x, y+1) - I(x, y)|
    v_grad = np.abs(np.diff(gray, axis=0))  # shape (h-1, w)

    # Horizontal boundary columns: positions where x % block_size == 0
    # (these are the left edges of each block)
    h_boundary_cols = np.arange(block_size - 1, w - 1, block_size)
    h_interior_mask = np.ones(w - 1, dtype=bool)
    h_interior_mask[h_boundary_cols] = False

    h_boundary_energy = float(np.mean(h_grad[:, h_boundary_cols])) if len(h_boundary_cols) > 0 else 0.0
    h_interior_energy = float(np.mean(h_grad[:, h_interior_mask])) if h_interior_mask.any() else 1.0

    # Vertical boundary rows
    v_boundary_rows = np.arange(block_size - 1, h - 1, block_size)
    v_interior_mask = np.ones(h - 1, dtype=bool)
    v_interior_mask[v_boundary_rows] = False

    v_boundary_energy = float(np.mean(v_grad[v_boundary_rows, :])) if len(v_boundary_rows) > 0 else 0.0
    v_interior_energy = float(np.mean(v_grad[v_interior_mask, :])) if v_interior_mask.any() else 1.0

    boundary_energy = (h_boundary_energy + v_boundary_energy) / 2.0
    interior_energy = (h_interior_energy + v_interior_energy) / 2.0

    return boundary_energy / (interior_energy + 1e-6)


def check_pixelation(
    frames: list[np.ndarray],
    config: PixelationConfig | None = None,
    timestamps: list[float] | None = None,
) -> CheckResult:
    """Run pixelation detection on sampled frames.

    Pass if non-pixelated frames >= min_good_ratio of total.
    """
    cfg = config or PixelationConfig()

    if not frames:
        return CheckResult(status="fail", metric_value=0.0, confidence=1.0,
                           details={"error": "no frames"})

    scores = []
    pixelated_frames = 0
    for i, frame in enumerate(frames):
        score = compute_blockiness(frame, cfg.block_size)
        scores.append(score)
        if score > cfg.pixelation_threshold:
            pixelated_frames += 1

    total = len(frames)
    good_ratio = (total - pixelated_frames) / total
    mean_score = float(np.mean(scores))
    status = "pass" if good_ratio >= cfg.min_good_ratio else "fail"

    return CheckResult(
        status=status,
        metric_value=round(good_ratio, 4),
        confidence=1.0,
        details={
            "mean_blockiness": round(mean_score, 4),
            "max_blockiness": round(float(np.max(scores)), 4),
            "min_blockiness": round(float(np.min(scores)), 4),
            "pixelated_frames": pixelated_frames,
            "total_frames": total,
            "good_ratio": round(good_ratio, 4),
            "min_good_ratio": cfg.min_good_ratio,
            "pixelation_threshold": cfg.pixelation_threshold,
            "block_size": cfg.block_size,
        },
    )
